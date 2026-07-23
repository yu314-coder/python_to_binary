from __future__ import annotations

import plistlib
import re
import struct
from pathlib import Path


_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_ICNS_TYPES = {
    16: b"icp4",
    32: b"icp5",
    64: b"icp6",
    128: b"ic07",
    256: b"ic08",
    512: b"ic09",
    1024: b"ic10",
}


class IconError(ValueError):
    pass


def _png_dimensions(data: bytes) -> tuple[int, int]:
    if len(data) < 24 or not data.startswith(_PNG_SIGNATURE) or data[12:16] != b"IHDR":
        raise IconError("icon image is not a PNG")
    return struct.unpack_from(">II", data, 16)


def _icns_record(png: bytes) -> bytes:
    width, height = _png_dimensions(png)
    if width != height or width not in _ICNS_TYPES:
        supported = ", ".join(str(size) for size in _ICNS_TYPES)
        raise IconError(f"PNG icon must be square and one of these sizes: {supported}")
    return _ICNS_TYPES[width] + struct.pack(">I", len(png) + 8) + png


def _pngs_from_ico(data: bytes) -> list[bytes]:
    if len(data) < 6:
        raise IconError("ICO header is truncated")
    reserved, image_type, count = struct.unpack_from("<HHH", data)
    if reserved != 0 or image_type != 1 or count == 0:
        raise IconError("file is not a Windows ICO")
    directory_end = 6 + count * 16
    if directory_end > len(data):
        raise IconError("ICO directory is truncated")

    images: list[bytes] = []
    for index in range(count):
        offset = 6 + index * 16
        width_byte, height_byte, _colors, _reserved, _planes, _bits, size, start = (
            struct.unpack_from("<BBBBHHII", data, offset)
        )
        width = width_byte or 256
        height = height_byte or 256
        end = start + size
        if start < directory_end or end > len(data):
            raise IconError(f"ICO image {index} points outside the file")
        image = data[start:end]
        if not image.startswith(_PNG_SIGNATURE):
            continue
        png_width, png_height = _png_dimensions(image)
        if (png_width, png_height) != (width, height):
            raise IconError(f"ICO image {index} has inconsistent dimensions")
        if width != height or width not in _ICNS_TYPES:
            # ICO permits sizes such as 48x48 that have no modern ICNS
            # record type. Preserve every compatible frame and skip only
            # the frames macOS cannot identify.
            continue
        images.append(image)
    if not images:
        raise IconError("ICO has no PNG frames; DIB-only ICO files are not supported yet")
    return images


def icon_to_icns(icon: Path) -> bytes:
    """Convert an ICNS, PNG, or PNG-backed ICO into ICNS bytes."""
    icon = icon.expanduser().resolve()
    if not icon.is_file():
        raise FileNotFoundError(f"icon does not exist: {icon}")
    data = icon.read_bytes()
    suffix = icon.suffix.lower()
    if suffix == ".icns":
        if len(data) < 8 or data[:4] != b"icns" or struct.unpack_from(">I", data, 4)[0] != len(data):
            raise IconError("ICNS file has an invalid header or size")
        return data
    if suffix == ".ico":
        pngs = _pngs_from_ico(data)
    elif suffix == ".png":
        pngs = [data]
    else:
        raise IconError("macOS app icons must be .icns, .ico, or .png")

    records: dict[bytes, bytes] = {}
    for png in pngs:
        record = _icns_record(png)
        records[record[:4]] = record
    payload = b"".join(records[key] for key in sorted(records))
    return b"icns" + struct.pack(">I", len(payload) + 8) + payload


def install_macos_icon(icon: Path | None, resources: Path) -> str | None:
    if icon is None:
        return None
    resources.mkdir(parents=True, exist_ok=True)
    filename = "AppIcon.icns"
    (resources / filename).write_bytes(icon_to_icns(icon))
    return filename


def macos_info_plist(name: str, executable: str, icon_filename: str | None = None) -> bytes:
    identifier_name = re.sub(r"[^A-Za-z0-9.-]+", "-", name).strip(".-").lower() or "app"
    values: dict[str, object] = {
        "CFBundleDisplayName": name,
        "CFBundleExecutable": executable,
        "CFBundleIdentifier": f"local.py2bin.{identifier_name}",
        "CFBundleName": name,
        "CFBundlePackageType": "APPL",
        "CFBundleVersion": "1",
        "CFBundleShortVersionString": "1.0",
        "NSHighResolutionCapable": True,
    }
    if icon_filename is not None:
        values["CFBundleIconFile"] = icon_filename
    return plistlib.dumps(values, fmt=plistlib.FMT_XML, sort_keys=True)
