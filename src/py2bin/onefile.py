from __future__ import annotations

import base64
import hashlib
import shlex
import shutil
import stat
import tarfile
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path

from .native.formats.pe import write_pe_shell_launcher
from .native.launcher import linux_shell_launcher, macos_shell_launcher
from .windows_icon import install_windows_identity


_MARKER_PREFIX = b"\nPY2BIN-ONEFILE-PAYLOAD-V1:"
_OFFSET_WIDTH = 20
_ZIP_COMPRESSLEVEL = 6


@dataclass(frozen=True, slots=True)
class OnefileResult:
    artifact: Path
    archive_bytes: int
    payload_sha256: str


def _payload_files(root: Path, exclude: "frozenset[Path]" = frozenset()) -> list[Path]:
    """Every file to pack, minus anything this packing is itself writing.

    An output written inside the directory being packed is an archive that
    contains itself, growing as it is read: one such run reached 217 GB before
    it was stopped. The output and the scratch directory beside it are left
    out, so a caller who puts the result in dist/ beside what it packs gets
    what they meant rather than a disk full.
    """
    skipped = {item.resolve() for item in exclude}
    kept = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        settled = path.resolve()
        if settled in skipped or any(
            parent in skipped for parent in settled.parents
        ):
            continue
        kept.append(path)
    return sorted(kept, key=lambda path: path.relative_to(root).as_posix())


def _zip_payload(root: Path, destination: Path) -> None:
    with zipfile.ZipFile(
        destination,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=_ZIP_COMPRESSLEVEL,
        allowZip64=True,
    ) as archive:
        for path in _payload_files(root, frozenset({destination})):
            relative = path.relative_to(root).as_posix()
            info = zipfile.ZipInfo.from_file(path, relative)
            info.compress_type = zipfile.ZIP_DEFLATED
            # A manually supplied ZipInfo otherwise falls back to zlib's
            # implicit level rather than the ZipFile-level setting.
            info._compresslevel = _ZIP_COMPRESSLEVEL
            with path.open("rb") as source, archive.open(info, "w", force_zip64=True) as output:
                while block := source.read(1024 * 1024):
                    output.write(block)


def _tar_payload(root: Path, destination: Path) -> None:
    with tarfile.open(
        destination,
        mode="w:gz",
        compresslevel=9,
        format=tarfile.PAX_FORMAT,
    ) as archive:
        for path in _payload_files(root, frozenset({destination})):
            archive.add(
                path,
                arcname=path.relative_to(root).as_posix(),
                recursive=False,
            )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _payload_marker(digest: str) -> bytes:
    return _MARKER_PREFIX + digest.encode("ascii") + b"\n"


def _fixed_assignment(name: str, value: int, *, powershell: bool = False) -> str:
    text = str(value)
    if len(text) > _OFFSET_WIDTH:
        raise ValueError("one-file payload exceeds the fixed launcher size field")
    prefix = "$" if powershell else ""
    return f"{prefix}{name}={text};" + (" " * (_OFFSET_WIDTH - len(text)))


def _posix_command(
    *,
    offset: int,
    length: int,
    digest: str,
    launcher: str,
) -> str:
    launcher_text = shlex.quote(launcher)
    # The cache is content-addressed. Extraction occurs under an atomic lock,
    # and a completion marker is written only after tar exits successfully.
    return (
        _fixed_assignment("O", offset)
        + _fixed_assignment("L", length)
        + "set -eu;umask 077;SELF=$0;"
        'case "$SELF" in /*) :;;*) SELF="$PWD/$SELF";;esac;'
        'if [ -n "${PY2BIN_CACHE_DIR:-}" ];then BASE=$PY2BIN_CACHE_DIR;'
        'elif [ -n "${XDG_CACHE_HOME:-}" ];then BASE="$XDG_CACHE_HOME/py2bin";'
        'else BASE="${HOME:-/tmp}/.cache/py2bin";fi;'
        f'CACHE="$BASE/{digest}";MARK="$CACHE/.py2bin-complete";'
        'mkdir -p "$BASE";'
        'if [ ! -f "$MARK" ];then LOCK="$CACHE.lock";'
        'if mkdir "$LOCK" 2>/dev/null;then '
        'TMP="$BASE/.extract-$$";'
        'trap \'rm -rf "$TMP";rmdir "$LOCK" 2>/dev/null||:\' 0 1 2 3 15;'
        'rm -rf "$TMP";mkdir -p "$TMP";'
        'tail -c +"$O" "$SELF"|head -c "$L"|'
        'tar -xzf - -C "$TMP";'
        'printf ok >"$TMP/.py2bin-complete";rm -rf "$CACHE";'
        'mv "$TMP" "$CACHE";rmdir "$LOCK";trap - 0 1 2 3 15;'
        'else while [ ! -f "$MARK" ];do sleep .05;done;fi;fi;'
        f'exec "$CACHE"/{launcher_text} "$@"'
    )


def _powershell_script(
    *,
    offset: int,
    digest: str,
    launcher: str,
    windowed: bool = False,
) -> str:
    launcher_ps = launcher.replace("'", "''").replace("/", "\\")
    mutex = f"Local\\py2bin_{digest}"
    # The handwritten launcher passes its Unicode path and original command
    # line through its child environment. This avoids two WMI/CIM queries on
    # every launch while preserving the argument suffix verbatim.
    return (
        _fixed_assignment("off", offset, powershell=True)
        + "$ErrorActionPreference='Stop';"
        # Without this, "Preparing modules for first use." is written to the
        # error stream, and PowerShell serialises that stream as CLIXML when it
        # is redirected. A user who redirects the program's output gets a page
        # of XML in front of it.
        "$ProgressPreference='SilentlyContinue';"
        # Everything below runs inside a try. Anything that goes wrong here -
        # a truncated payload, a full disk, a launcher missing from the archive
        # - used to end the process with no message at all, because the error
        # went to the error stream as CLIXML and looked like noise. The message
        # is written to stdout instead, which is not serialised.
        "try{"
        "$s=$env:PY2BIN_ONEFILE_SELF;$c=$env:PY2BIN_ONEFILE_COMMAND;"
        "$env:PY2BIN_ONEFILE_SELF=$null;$env:PY2BIN_ONEFILE_COMMAND=$null;"
        "if(!$s -or !$c){throw 'one-file launcher environment is missing'};"
        "$raw='';"
        "if($c.StartsWith('\"')){$q=$c.IndexOf('\"',1);"
        "if($q -ge 0){$raw=$c.Substring($q+1).TrimStart()}}"
        "else{$q=$c.IndexOf(' ');if($q -lt 0){$q=$c.IndexOf(\"`t\")};"
        "if($q -ge 0){$raw=$c.Substring($q+1).TrimStart()}};"
        "if($env:PY2BIN_CACHE_DIR){$b=$env:PY2BIN_CACHE_DIR}"
        "else{$b=[IO.Path]::Combine($env:LOCALAPPDATA,'py2bin')};"
        f"$r=[IO.Path]::Combine($b,'{digest}');"
        "$m=[IO.Path]::Combine($r,'.py2bin-complete');"
        "if(![IO.File]::Exists($m)){"
        f"$mx=[Threading.Mutex]::new($false,'{mutex}');"
        "$null=$mx.WaitOne();"
        "try{if(![IO.File]::Exists($m)){"
        "[IO.Directory]::CreateDirectory($b)|Out-Null;"
        "$t=[IO.Path]::Combine($b,'.extract.'+$PID);"
        "if([IO.Directory]::Exists($t)){[IO.Directory]::Delete($t,$true)};"
        "[IO.Directory]::CreateDirectory($t)|Out-Null;"
        "$z=[IO.Path]::Combine($t,'payload.zip');"
        "$i=[IO.File]::OpenRead($s);"
        "try{$i.Position=$off;$o=[IO.File]::Create($z);"
        "try{$i.CopyTo($o)}finally{$o.Dispose()}}finally{$i.Dispose()};"
        "Add-Type -AssemblyName System.IO.Compression.FileSystem;"
        "[IO.Compression.ZipFile]::ExtractToDirectory($z,$t);"
        "[IO.File]::Delete($z);"
        "[IO.File]::WriteAllText([IO.Path]::Combine($t,'.py2bin-complete'),'ok');"
        "if([IO.Directory]::Exists($r)){[IO.Directory]::Delete($r,$true)};"
        "[IO.Directory]::Move($t,$r)"
        "}}finally{$mx.ReleaseMutex();$mx.Dispose()}};"
        "$si=[Diagnostics.ProcessStartInfo]::new();"
        f"$si.FileName=[IO.Path]::Combine($r,'{launcher_ps}');"
        "$si.Arguments=$raw;$si.UseShellExecute=$false;"
        # Only a windowed build wants this. For a console program it is
        # the same mistake the launcher stub made one level up: a child
        # denied a console has nowhere to write, so the program runs, its
        # output is discarded, and it exits 0 having printed nothing.
        f"$si.CreateNoWindow=${str(windowed).lower()};"
        "$child=[Diagnostics.Process]::Start($si);$child.WaitForExit();"
        "$code=$child.ExitCode"
        # `exit` is left outside the try: inside one it unwinds through a flow
        # exception rather than a normal one, which is a needless thing to have
        # to reason about next to a catch.
        "}catch{"
        "[Console]::Out.WriteLine('py2bin one-file launcher failed: '"
        "+$_.Exception.Message);"
        "exit 3};"
        "exit $code"
    )


def _powershell_prefix(script: str) -> bytes:
    encoded = base64.b64encode(script.encode("utf-16-le")).decode("ascii")
    return (
        "powershell.exe -NoLogo -NoProfile -NonInteractive "
        "-ExecutionPolicy Bypass -EncodedCommand "
        + encoded
        + " "
    ).encode("ascii")


def _windows_stub(
    command_prefix: bytes,
    *,
    machine: str,
    name: str,
    icon: Path | None,
    temporary_root: Path,
    windowed: bool,
) -> bytes:
    image = write_pe_shell_launcher(
        command_prefix,
        machine,
        windowed=windowed,
    )
    with tempfile.TemporaryDirectory(
        prefix="py2bin-icon-", dir=temporary_root
    ) as directory:
        executable = Path(directory) / "launcher.exe"
        executable.write_bytes(image)
        install_windows_identity(
            executable,
            name,
            version="1.0.0.0",
            icon=icon,
        )
        return executable.read_bytes()


def _write_executable(path: Path, image: bytes) -> None:
    path.write_bytes(image)
    path.chmod(
        path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH
    )


def _write_executable_parts(
    path: Path,
    prefix: bytes,
    marker: bytes,
    archive: Path,
) -> None:
    with path.open("wb") as output:
        output.write(prefix)
        output.write(marker)
        with archive.open("rb") as payload:
            shutil.copyfileobj(payload, output, 1024 * 1024)
    path.chmod(
        path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH
    )


def create_onefile(
    payload_root: Path,
    output: Path,
    *,
    target: str,
    launcher: Path,
    icon: Path | None = None,
    info_plist: bytes | None = None,
    code_resources: bytes | None = None,
    windows_windowed: bool = False,
) -> OnefileResult:
    """Create a self-extracting compatibility executable.

    The launcher is handwritten PE/ELF/Mach-O machine code. The embedded
    application remains a target-specific CPython runtime bundle.
    """

    payload_root = payload_root.resolve()
    output = output.resolve()
    if output.parent == payload_root or payload_root in output.parents:
        raise ValueError(
            f"the single file would be written inside what it packs:\n"
            f"  packing {payload_root}\n"
            f"  writing {output}\n"
            f"An archive that contains itself grows as it is read. Write it "
            f"somewhere else."
        )
    relative_launcher = launcher.resolve().relative_to(payload_root).as_posix()
    windows_name = Path(relative_launcher).stem
    windows = target.startswith("windows-")
    machine = target.rpartition("-")[2]
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="py2bin-onefile-", dir=output.parent
    ) as directory:
        archive_path = Path(directory) / (
            "payload.zip" if windows else "payload.tar.gz"
        )
        if windows:
            _zip_payload(payload_root, archive_path)
        else:
            _tar_payload(payload_root, archive_path)
        archive_size = archive_path.stat().st_size
        digest = _file_sha256(archive_path)
        marker = _payload_marker(digest)

        if windows:
            placeholder = _powershell_prefix(
                _powershell_script(
                    offset=0,
                    digest=digest,
                    launcher=relative_launcher,
                    windowed=windows_windowed,
                )
            )
            stub = _windows_stub(
                placeholder,
                machine=machine,
                name=windows_name,
                icon=icon,
                temporary_root=output.parent,
                windowed=windows_windowed,
            )
            offset = len(stub) + len(marker)
            command = _powershell_prefix(
                _powershell_script(
                    offset=offset,
                    digest=digest,
                    launcher=relative_launcher,
                    windowed=windows_windowed,
                )
            )
            final_stub = _windows_stub(
                command,
                machine=machine,
                name=windows_name,
                icon=icon,
                temporary_root=output.parent,
                windowed=windows_windowed,
            )
            if len(final_stub) != len(stub):
                raise AssertionError(
                    "Windows one-file launcher size changed after patching"
                )
            _write_executable_parts(output, final_stub, marker, archive_path)
        elif target.startswith("linux-"):
            placeholder_command = _posix_command(
                offset=0,
                length=archive_size,
                digest=digest,
                launcher=relative_launcher,
            )
            stub = linux_shell_launcher(placeholder_command, machine)
            # tail -c +N uses a one-based byte position.
            offset = len(stub) + len(marker) + 1
            command = _posix_command(
                offset=offset,
                length=archive_size,
                digest=digest,
                launcher=relative_launcher,
            )
            final_stub = linux_shell_launcher(command, machine)
            if len(final_stub) != len(stub):
                raise AssertionError(
                    "Linux one-file launcher size changed after patching"
                )
            _write_executable_parts(output, final_stub, marker, archive_path)
        else:
            # The ARM64 Mach-O ad-hoc signature must cover the embedded payload,
            # so the current Mach-O writer consumes it as part of __TEXT.
            archive = archive_path.read_bytes()
            extra = marker + archive
            placeholder_command = _posix_command(
                offset=0,
                length=archive_size,
                digest=digest,
                launcher=relative_launcher,
            )
            placeholder_image = macos_shell_launcher(
                placeholder_command,
                machine,
                info_plist,
                code_resources,
                extra,
            )
            marker_at = placeholder_image.find(marker)
            if marker_at < 0:
                raise AssertionError(
                    "one-file marker is missing from native launcher"
                )
            offset = marker_at + len(marker) + 1
            command = _posix_command(
                offset=offset,
                length=archive_size,
                digest=digest,
                launcher=relative_launcher,
            )
            image = macos_shell_launcher(
                command,
                machine,
                info_plist,
                code_resources,
                extra,
            )
            final_marker_at = image.find(marker)
            if final_marker_at + len(marker) + 1 != offset:
                raise AssertionError(
                    "POSIX one-file launcher offset changed after patching"
                )
            _write_executable(output, image)

    return OnefileResult(output, archive_size, digest)


def pack_app_bundle(
    bundle: Path, target: str, name: str | None = None
) -> tuple[int, int]:
    """Fold a finished `.app` into an `.app` holding one real file.

    A macOS application is a directory, and it has to stay one - the launcher
    Finder runs is `Contents/MacOS/<name>`, and Gatekeeper reads
    `Contents/Info.plist` beside it. So this does not produce a single file
    on disk; it produces a bundle whose payload has become one. Five hundred
    files turn into two, and what the user drags is the same shape they
    already know.

    The packed bundle is the sealed one, byte for byte: the payload is written
    after the seal, so what unpacks at run time is the application that would
    have shipped, signature included.

    What guards the payload is the launcher's own signature, not the digest in
    its script. That digest names the cache directory and nothing more - it is
    taken at build time, so a payload edited afterwards would simply unpack
    under the name of the payload that should have been there. On arm64 the
    archive is carried inside `__TEXT`, which the ad-hoc signature covers, so
    the kernel refuses to start a binary whose payload has been altered: a
    flipped byte in the archive ends in SIGKILL before any of this script
    runs. That is measured, not assumed.

    Answers how many files were folded in, and the size of the bundle that
    results.
    """

    bundle = bundle.resolve()
    contents = bundle / "Contents"
    plist = contents / "Info.plist"
    if not plist.is_file():
        raise ValueError(f"{bundle} has no Contents/Info.plist to pack")
    executables = [
        item for item in (contents / "MacOS").iterdir() if item.is_file()
    ]
    inner = next(
        (item for item in executables if item.name == (name or bundle.stem)),
        None,
    ) or (executables[0] if executables else None)
    if inner is None:
        raise ValueError(f"{bundle} has no executable in Contents/MacOS")

    held = sum(1 for item in bundle.rglob("*") if item.is_file())
    with tempfile.TemporaryDirectory(
        prefix="py2bin-app-onefile-", dir=bundle.parent
    ) as scratch:
        room = Path(scratch)
        # Packed from a copy: `create_onefile` refuses to write inside what it
        # reads, and the file it writes is going where the original stands.
        packed_from = room / bundle.name
        shutil.copytree(bundle, packed_from, symlinks=True)
        launcher = packed_from / inner.relative_to(bundle)
        single = room / "packed"
        create_onefile(
            payload_root=packed_from,
            output=single,
            target=target,
            launcher=launcher,
            info_plist=plist.read_bytes(),
        )
        # The old tree goes only once the new executable exists, so a failure
        # anywhere above leaves the bundle that was already built.
        shutil.rmtree(bundle)
        (bundle / "Contents" / "MacOS").mkdir(parents=True)
        shutil.copy2(single, bundle / "Contents" / "MacOS" / inner.name)
        (bundle / "Contents" / "MacOS" / inner.name).chmod(0o755)
        shutil.copy2(
            packed_from / "Contents" / "Info.plist",
            bundle / "Contents" / "Info.plist",
        )

    # Sealed again, over the two files that are left. The launcher's signature
    # names a `CodeResources` that described the bundle before it was packed,
    # and `codesign` says so - "code has no resources but signature indicates
    # they must be present" - which is a bundle macOS would come to distrust.
    # Re-sealing writes the manifest for what is actually there.
    from .macos_seal import seal

    seal(bundle)
    total = sum(item.stat().st_size for item in bundle.rglob("*") if item.is_file())
    return held, total


def pack_beside_executable(
    program: Path, target: str, carried: list[str]
) -> tuple[int, int]:
    """Fold a program and what it carries into one self-extracting file.

    A target with no bundle - Linux, or a bare Windows executable - leaves its
    packages and its data directories sitting beside the program, and all of
    them have to travel together or none of it runs. This makes them one file:
    the launcher unpacks the lot into a content-addressed cache the first time
    it runs and executes the program from in there, so the directories are
    beside it again exactly as the build left them.

    Only what the program was given is packed. The generated C and the
    fetch scratch are build leavings, and a payload that carried them would
    unpack a copy of the compiler's workings next to the application.

    Answers how many files were folded in, and the size of the file that
    results.
    """

    program = program.resolve()
    if not program.is_file():
        raise ValueError(f"{program} is not a file to pack")
    beside = program.parent
    wanted = [program]
    for name in dict.fromkeys(carried):
        item = beside / name
        if item.exists():
            wanted.append(item)

    with tempfile.TemporaryDirectory(
        prefix="py2bin-beside-", dir=beside
    ) as scratch:
        room = Path(scratch)
        # Staged rather than packed in place: `create_onefile` reads the whole
        # directory it is given, and the directory the program sits in also
        # holds the C it was compiled from.
        staged = room / "payload"
        staged.mkdir()
        held = 0
        for item in wanted:
            landing = staged / item.name
            if item.is_dir():
                shutil.copytree(item, landing, symlinks=True)
                held += sum(1 for f in landing.rglob("*") if f.is_file())
            else:
                shutil.copy2(item, landing)
                held += 1
        single = room / "packed"
        create_onefile(
            payload_root=staged,
            output=single,
            target=target,
            launcher=staged / program.name,
        )
        # Only the program is replaced. What sat beside it is left exactly
        # where it was, because there is no way from here to tell a directory
        # this build copied in from one the *user* keeps there - `--include
        # web` run in a source tree carries a directory that is already
        # beside the output, and deleting it after packing destroys the
        # original. The loose copies are redundant once the single file
        # exists; which of them to remove is the caller's to decide, and it
        # knows what it made.
        image = single.read_bytes()
        program.write_bytes(image)
        program.chmod(0o755)

    return held, program.stat().st_size
