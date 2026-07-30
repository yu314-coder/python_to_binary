"""Seal a finished .app the way macOS expects to find it sealed.

The seal over a bundle's contents cannot be written while the bundle is being
built, because at that point almost nothing is in it: the interpreter, the
packages and the program's own files all arrive afterwards. Writing it early
and naming the late arrivals in omit rules does not work either - a seal that
excuses files from being checked is the one thing strict validation refuses to
accept, and it refuses it whatever the files are.

So this runs last, over the bundle as it will actually ship, and hashes what
is there. Nothing is omitted beyond what Apple's own default rules omit.

A signature covers its own resource seal: the code directory carries the hash
of CodeResources in a slot of its own. Re-sealing therefore invalidates the
signature that was written earlier, and the main executable has to be signed
again afterwards - which is why sealing and signing happen together here
rather than in two places that could disagree.
"""

from __future__ import annotations

import hashlib
import plistlib
import struct
from pathlib import Path

#: Apple's default resource rules, verbatim. The omit entries here are the
#: ones codesign writes itself, and are not what "custom omit rules" means.
_RULES = {
    "^Resources/": True,
    "^Resources/.*\\.lproj/": {"optional": True, "weight": 1000},
    "^Resources/.*\\.lproj/locversion.plist$": {"omit": True, "weight": 1100},
    "^Resources/Base\\.lproj/": {"weight": 1010},
    "^version.plist$": True,
}

_RULES2 = {
    ".*\\.dSYM($|/)": {"weight": 11},
    "^(.*/)?\\.DS_Store$": {"omit": True, "weight": 2000},
    # Frameworks are deliberately absent from this group. Marking a
    # framework as nested code tells macOS to stop at its identity and
    # validate it on its own terms - and the interpreter carried here is
    # pruned and thinned after its author signed it, so on its own terms it
    # no longer verifies. Sealed as ordinary contents instead, every one of
    # its files is hashed by the bundle that actually ships it.
    "^(SharedFrameworks|PlugIns|Plug-ins|XPCServices|Helpers)/": {
        "nested": True,
        "weight": 10,
    },
    "^.*": True,
    "^Info\\.plist$": {"omit": True, "weight": 20},
    "^PkgInfo$": {"omit": True, "weight": 20},
    "^Resources/": {"weight": 20},
    "^Resources/.*\\.lproj/": {"optional": True, "weight": 1000},
    "^Resources/.*\\.lproj/locversion.plist$": {"omit": True, "weight": 1100},
    "^Resources/Base\\.lproj/": {"weight": 1010},
    "^embedded\\.provisionprofile$": {"weight": 20},
    "^version\\.plist$": {"weight": 20},
}

_MACHO_MAGIC = (0xFEEDFACE, 0xFEEDFACF)
_FAT_MAGIC = (0xCAFEBABE, 0xCAFEBABF)
_LC_CODE_SIGNATURE = 0x1D
_LC_SEGMENT_64 = 0x19


def _align(value: int, alignment: int = 0x4000) -> int:
    """Round up to a page, for the segment sizes macOS maps."""
    return (value + alignment - 1) & ~(alignment - 1)


class SealError(Exception):
    """A bundle could not be sealed."""


def is_macho(path: Path) -> bool:
    """Whether a file begins with a Mach-O or universal header."""
    try:
        with path.open("rb") as handle:
            head = handle.read(4)
    except OSError:
        return False
    if len(head) < 4:
        return False
    return (
        struct.unpack(">I", head)[0] in _FAT_MAGIC
        or struct.unpack("<I", head)[0] in _MACHO_MAGIC
    )


def code_directory_hash(image: bytes) -> bytes | None:
    """The cdhash of a signed Mach-O: its code directory, hashed, truncated.

    This is the identity macOS records for nested code, and it is read out of
    the signature already present rather than recomputed, so a framework whose
    binary this build never touched keeps the identity its author gave it.
    """
    signature = _signature_span(image)
    if signature is None:
        return None
    start, size = signature
    blob = image[start:start + size]
    if len(blob) < 12:
        return None
    magic, _length, count = struct.unpack_from(">III", blob, 0)
    if magic != 0xFADE0CC0:
        return None
    for index in range(count):
        entry = 12 + index * 8
        if entry + 8 > len(blob):
            break
        _slot, offset = struct.unpack_from(">II", blob, entry)
        if offset + 8 > len(blob):
            continue
        kind, length = struct.unpack_from(">II", blob, offset)
        if kind == 0xFADE0C02:  # the code directory
            return hashlib.sha256(blob[offset:offset + length]).digest()[:20]
    return None


def _signature_span(image: bytes) -> tuple[int, int] | None:
    """Where LC_CODE_SIGNATURE points, for a thin little-endian image."""
    if len(image) < 32 or struct.unpack_from("<I", image, 0)[0] not in _MACHO_MAGIC:
        return None
    count = struct.unpack_from("<I", image, 16)[0]
    offset = 32
    for _ in range(count):
        if offset + 8 > len(image):
            return None
        command, size = struct.unpack_from("<II", image, offset)
        if size <= 0:
            return None
        if command == _LC_CODE_SIGNATURE:
            start, length = struct.unpack_from("<II", image, offset + 8)
            return start, length
        offset += size
    return None


def _relative_paths(contents: Path) -> list[Path]:
    """Every file under Contents, deepest last, signature directory aside."""
    found = []
    for path in sorted(contents.rglob("*")):
        if path.is_dir() and not path.is_symlink():
            continue
        relative = path.relative_to(contents)
        if relative.parts and relative.parts[0] == "_CodeSignature":
            continue
        found.append(relative)
    return found


def _nested_bundles(contents: Path) -> list[Path]:
    """Code bundles inside this one, which are sealed by identity not content."""
    nested = []
    for parent in ("SharedFrameworks", "PlugIns", "XPCServices", "Helpers"):
        directory = contents / parent
        if not directory.is_dir():
            continue
        for entry in sorted(directory.iterdir()):
            if entry.is_dir() and entry.suffix in (".framework", ".app", ".bundle", ".xpc"):
                nested.append(entry.relative_to(contents))
    return nested


def _bundle_executable(bundle: Path) -> str | None:
    try:
        info = plistlib.loads((bundle / "Contents" / "Info.plist").read_bytes())
    except (OSError, ValueError):
        return None
    name = info.get("CFBundleExecutable")
    return name if isinstance(name, str) else None


def _framework_binary(contents: Path, relative: Path) -> Path | None:
    """The Mach-O a nested framework or app is identified by."""
    root = contents / relative
    if relative.suffix == ".framework":
        current = root / "Versions" / "Current" / root.stem
        if current.is_file():
            return current
        for version in sorted((root / "Versions").glob("*")) if (root / "Versions").is_dir() else []:
            candidate = version / root.stem
            if candidate.is_file():
                return candidate
        direct = root / root.stem
        return direct if direct.is_file() else None
    name = _bundle_executable(root)
    if name:
        candidate = root / "Contents" / "MacOS" / name
        if candidate.is_file():
            return candidate
    return None


def code_resources(bundle: Path) -> bytes:
    """Build the seal for a finished bundle, hashing what it actually holds."""
    contents = bundle / "Contents"
    if not contents.is_dir():
        raise SealError(f"not an app bundle: {bundle}")
    executable = _bundle_executable(bundle)
    skip = {Path("Info.plist"), Path("PkgInfo")}
    if executable:
        skip.add(Path("MacOS") / executable)

    nested = _nested_bundles(contents)
    files: dict[str, bytes] = {}
    files2: dict[str, dict[str, object]] = {}

    for relative in nested:
        binary = _framework_binary(contents, relative)
        if binary is None:
            continue
        digest = code_directory_hash(binary.read_bytes())
        if digest is None:
            continue
        files2[relative.as_posix()] = {"cdhash": digest}

    for relative in _relative_paths(contents):
        if relative in skip:
            continue
        # Anything inside a nested bundle is covered by that bundle's own
        # identity, and naming it again here would seal it twice.
        if any(relative.is_relative_to(entry) for entry in nested):
            continue
        path = contents / relative
        if path.is_symlink():
            files2[relative.as_posix()] = {"symlink": str(path.readlink())}
            continue
        try:
            payload = path.read_bytes()
        except OSError as error:
            raise SealError(f"cannot read {path}: {error}") from error
        files2[relative.as_posix()] = {"hash2": hashlib.sha256(payload).digest()}
        if relative.parts and relative.parts[0] == "Resources":
            files[relative.as_posix()] = hashlib.sha1(payload).digest()

    return plistlib.dumps(
        {"files": files, "files2": files2, "rules": _RULES, "rules2": _RULES2},
        fmt=plistlib.FMT_XML,
        sort_keys=True,
    )


def resign(binary: Path, info_plist: bytes | None = None,
           resources: bytes | None = None) -> None:
    """Sign an already-written Mach-O again, over a seal it did not have.

    The image keeps its layout; only the signature at the end is replaced, and
    the load command that describes it is corrected to the new length.
    """
    from .native.formats.macho import _adhoc_signature

    image = bytearray(binary.read_bytes())
    span = _signature_span(bytes(image))
    if span is None:
        raise SealError(f"{binary} carries no signature to replace")
    signature_offset, _old = span

    exec_base = exec_size = 0
    linkedit = None
    count = struct.unpack_from("<I", image, 16)[0]
    offset = 32
    for _ in range(count):
        command, size = struct.unpack_from("<II", image, offset)
        if command == _LC_SEGMENT_64:
            name = bytes(image[offset + 8:offset + 24]).rstrip(b"\0")
            if name == b"__TEXT":
                exec_base, exec_size = struct.unpack_from("<QQ", image, offset + 24)
            elif name == b"__LINKEDIT":
                linkedit = offset
        offset += size

    def build() -> bytes:
        signature = _adhoc_signature(
            bytes(image[:signature_offset]),
            signature_offset,
            exec_base,
            exec_size,
            info_plist=info_plist,
            code_resources=resources,
        )
        return signature + b"\0" * (-len(signature) % 16)

    # The load command that says how long the signature is, and the segment
    # sizes that cover it, sit below the signature and are therefore hashed by
    # it. They have to hold their final values before the hashing happens, so
    # the length is worked out first and the image is only signed once it is
    # already describing itself correctly. Signing first and correcting
    # afterwards invalidates exactly the hashes just written.
    length = len(build())
    offset = 32
    for _ in range(count):
        command, size = struct.unpack_from("<II", image, offset)
        if command == _LC_CODE_SIGNATURE:
            struct.pack_into("<II", image, offset + 8, signature_offset, length)
        offset += size
    if linkedit is not None:
        vmaddr, _vmsize, fileoff, _filesize = struct.unpack_from("<QQQQ", image, linkedit + 24)
        span = signature_offset + length - fileoff
        struct.pack_into(
            "<QQQQ", image, linkedit + 24, vmaddr, _align(span), fileoff, span
        )

    padded = build()
    if len(padded) != length:
        raise SealError("signature length did not settle")
    del image[signature_offset:]
    image.extend(padded)

    binary.write_bytes(bytes(image))
    binary.chmod(binary.stat().st_mode | 0o111)


def _drop_stale_seals(directory: Path) -> int:
    """Remove signatures inside frameworks this build has already altered.

    Pruning a framework's standard library invalidates the seal its author
    wrote over it, and leaving that seal in place means shipping a signature
    that cannot be satisfied. The binary keeps its own signature - that is
    what the kernel checks to run it - but the resource seal it no longer
    matches goes.
    """
    removed = 0
    if not directory.is_dir():
        return 0
    for signature in sorted(directory.rglob("_CodeSignature")):
        if signature.is_dir():
            for entry in sorted(signature.rglob("*"), reverse=True):
                entry.unlink() if entry.is_file() or entry.is_symlink() else entry.rmdir()
            signature.rmdir()
            removed += 1
    return removed


def resign_libraries(bundle: Path) -> int:
    """Sign every library in the bundle again, over its contents as shipped.

    Relinking an extension so it finds the copy of OpenSSL carried here edits
    its load commands, and that leaves the signature its author wrote no longer
    describing it. Nothing complains while building, and nothing complains on a
    machine with System Integrity Protection turned off. On a stock Mac dyld
    maps a page, the hash does not match, and the kernel kills the process:

        EXC_BAD_ACCESS (SIGKILL (Code Signature Invalid))
        Namespace CODESIGNING, Code 2, Invalid Page

    Every Mach-O here is signed again rather than only the ones known to have
    been edited, because a signature that is already good stays good, and a
    list of which files were touched is the kind of thing that goes stale.
    """
    count = 0
    executable = _bundle_executable(bundle)
    main = (bundle / "Contents" / "MacOS" / executable) if executable else None
    for path in sorted((bundle / "Contents").rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        if main is not None and path == main:
            continue  # signed later, over the resource seal
        if path.suffix not in (".so", ".dylib") and not is_macho(path):
            continue
        try:
            resign(path)
            count += 1
        except (SealError, OSError, struct.error):
            # Not something this can sign - a fat file, or one with no
            # signature to replace. Left exactly as it was.
            continue
    return count


def seal(bundle: Path) -> int:
    """Seal a finished bundle and sign its executable over that seal.

    Returns the number of entries sealed.
    """
    _drop_stale_seals(bundle / "Contents" / "Frameworks")
    signed = resign_libraries(bundle)
    resources = code_resources(bundle)
    signature_directory = bundle / "Contents" / "_CodeSignature"
    signature_directory.mkdir(parents=True, exist_ok=True)
    (signature_directory / "CodeResources").write_bytes(resources)

    executable = _bundle_executable(bundle)
    if executable:
        binary = bundle / "Contents" / "MacOS" / executable
        if binary.is_file() and is_macho(binary):
            info = (bundle / "Contents" / "Info.plist").read_bytes()
            resign(binary, info, resources)
    return len(plistlib.loads(resources)["files2"])
