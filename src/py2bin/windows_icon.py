from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path


RT_ICON = 3
RT_GROUP_ICON = 14


@dataclass(frozen=True, slots=True)
class ResourceBlob:
    data: bytes
    codepage: int = 0


def _align(value: int, alignment: int) -> int:
    return (value + alignment - 1) & -alignment


def _pe_layout(image: bytes) -> dict[str, object]:
    if len(image) < 0x100 or image[:2] != b"MZ":
        raise ValueError("Windows icon target is not a PE executable")
    pe_offset = struct.unpack_from("<I", image, 0x3C)[0]
    if image[pe_offset : pe_offset + 4] != b"PE\0\0":
        raise ValueError("Windows icon target has no PE signature")
    coff = pe_offset + 4
    section_count = struct.unpack_from("<H", image, coff + 2)[0]
    optional_size = struct.unpack_from("<H", image, coff + 16)[0]
    optional = coff + 20
    if struct.unpack_from("<H", image, optional)[0] != 0x20B:
        raise ValueError("Windows icon target must be PE32+")
    section_alignment, file_alignment = struct.unpack_from(
        "<II", image, optional + 32
    )
    size_of_headers = struct.unpack_from("<I", image, optional + 60)[0]
    sections_offset = optional + optional_size
    sections: list[tuple[int, int, int, int]] = []
    for index in range(section_count):
        offset = sections_offset + index * 40
        virtual_size, virtual_address, raw_size, raw_offset = struct.unpack_from(
            "<IIII", image, offset + 8
        )
        sections.append((virtual_address, virtual_size, raw_offset, raw_size))
    return {
        "pe": pe_offset,
        "coff": coff,
        "optional": optional,
        "sections_offset": sections_offset,
        "section_count": section_count,
        "section_alignment": section_alignment,
        "file_alignment": file_alignment,
        "size_of_headers": size_of_headers,
        "sections": sections,
    }


def _rva_to_offset(
    rva: int, sections: list[tuple[int, int, int, int]]
) -> int:
    for virtual_address, virtual_size, raw_offset, raw_size in sections:
        extent = max(virtual_size, raw_size)
        if virtual_address <= rva < virtual_address + extent:
            return raw_offset + (rva - virtual_address)
    raise ValueError(f"PE resource RVA 0x{rva:x} is outside every section")


def _existing_resources(
    image: bytes, layout: dict[str, object]
) -> dict[tuple[int, int, int], ResourceBlob]:
    optional = int(layout["optional"])
    resource_rva, resource_size = struct.unpack_from("<II", image, optional + 112 + 16)
    if not resource_rva or not resource_size:
        return {}
    sections = list(layout["sections"])
    resource_offset = _rva_to_offset(resource_rva, sections)
    resources: dict[tuple[int, int, int], ResourceBlob] = {}

    def walk(directory_offset: int, identifiers: tuple[int, ...]) -> None:
        absolute = resource_offset + directory_offset
        if absolute + 16 > len(image):
            raise ValueError("PE resource directory is truncated")
        named, numeric = struct.unpack_from("<HH", image, absolute + 12)
        count = named + numeric
        for index in range(count):
            entry = absolute + 16 + index * 8
            if entry + 8 > len(image):
                raise ValueError("PE resource entry is truncated")
            name, child = struct.unpack_from("<II", image, entry)
            if name & 0x80000000:
                # Python's executable resources use numeric identifiers. Skip
                # an unfamiliar named leaf rather than mis-encoding its name.
                continue
            identifier = name & 0xFFFF
            if child & 0x80000000:
                walk(child & 0x7FFFFFFF, identifiers + (identifier,))
                continue
            data_entry = resource_offset + child
            if data_entry + 16 > len(image):
                raise ValueError("PE resource data entry is truncated")
            data_rva, size, codepage, _reserved = struct.unpack_from(
                "<IIII", image, data_entry
            )
            data_offset = _rva_to_offset(data_rva, sections)
            if data_offset + size > len(image):
                raise ValueError("PE resource data points outside the executable")
            if len(identifiers) == 2:
                key = (identifiers[0], identifiers[1], identifier)
                resources[key] = ResourceBlob(
                    image[data_offset : data_offset + size], codepage
                )

    walk(0, ())
    return resources


def _ico_resources(icon: bytes) -> dict[tuple[int, int, int], ResourceBlob]:
    if len(icon) < 6:
        raise ValueError("ICO header is truncated")
    reserved, image_type, count = struct.unpack_from("<HHH", icon)
    if reserved != 0 or image_type != 1 or not count:
        raise ValueError("Windows executable icon must be a non-empty ICO file")
    if 6 + count * 16 > len(icon):
        raise ValueError("ICO directory is truncated")
    group = bytearray(struct.pack("<HHH", 0, 1, count))
    resources: dict[tuple[int, int, int], ResourceBlob] = {}
    for index in range(count):
        entry = struct.unpack_from("<BBBBHHII", icon, 6 + index * 16)
        width, height, colors, reserved_byte, planes, bits, size, offset = entry
        if offset < 6 + count * 16 or offset + size > len(icon):
            raise ValueError(f"ICO frame {index} points outside the file")
        resource_id = index + 1
        resources[(RT_ICON, resource_id, 0x0409)] = ResourceBlob(
            icon[offset : offset + size]
        )
        group.extend(
            struct.pack(
                "<BBBBHHIH",
                width,
                height,
                colors,
                reserved_byte,
                planes,
                bits,
                size,
                resource_id,
            )
        )
    resources[(RT_GROUP_ICON, 1, 0x0409)] = ResourceBlob(bytes(group))
    return resources


def _resource_section(
    resources: dict[tuple[int, int, int], ResourceBlob],
    section_rva: int,
) -> bytes:
    tree: dict[int, dict[int, dict[int, ResourceBlob]]] = {}
    for (resource_type, name, language), blob in resources.items():
        tree.setdefault(resource_type, {}).setdefault(name, {})[language] = blob
    output = bytearray()
    data_entries: list[tuple[int, ResourceBlob]] = []

    def emit_directory(node, depth: int) -> int:
        offset = len(output)
        items = sorted(node.items())
        output.extend(struct.pack("<IIHHHH", 0, 0, 0, 0, 0, len(items)))
        entries_offset = len(output)
        output.extend(b"\0" * (8 * len(items)))
        for index, (identifier, child) in enumerate(items):
            if depth < 2:
                child_offset = emit_directory(child, depth + 1)
                target = 0x80000000 | child_offset
            else:
                data_entry = len(output)
                output.extend(b"\0" * 16)
                data_entries.append((data_entry, child))
                target = data_entry
            struct.pack_into(
                "<II", output, entries_offset + index * 8, identifier, target
            )
        return offset

    emit_directory(tree, 0)
    while len(output) % 4:
        output.append(0)
    for data_entry, blob in data_entries:
        data_offset = len(output)
        output.extend(blob.data)
        while len(output) % 4:
            output.append(0)
        struct.pack_into(
            "<IIII",
            output,
            data_entry,
            section_rva + data_offset,
            len(blob.data),
            blob.codepage,
            0,
        )
    return bytes(output)


def install_windows_icon(executable: Path, icon: Path) -> None:
    """Replace PE icon resources without invoking a Windows resource compiler."""

    executable = executable.expanduser().resolve()
    icon = icon.expanduser().resolve()
    if icon.suffix.lower() != ".ico":
        raise ValueError("Windows executable icons must be .ico files")
    image = bytearray(executable.read_bytes())
    layout = _pe_layout(image)
    resources = {
        key: value
        for key, value in _existing_resources(image, layout).items()
        if key[0] not in {RT_ICON, RT_GROUP_ICON}
    }
    resources.update(_ico_resources(icon.read_bytes()))

    sections = list(layout["sections"])
    section_alignment = int(layout["section_alignment"])
    file_alignment = int(layout["file_alignment"])
    section_count = int(layout["section_count"])
    sections_offset = int(layout["sections_offset"])
    size_of_headers = int(layout["size_of_headers"])
    header_offset = sections_offset + section_count * 40
    if header_offset + 40 > size_of_headers:
        raise ValueError("PE headers have no room for an icon resource section")
    section_rva = _align(
        max(rva + max(virtual_size, raw_size) for rva, virtual_size, _raw, raw_size in sections),
        section_alignment,
    )
    section_data = _resource_section(resources, section_rva)
    raw_offset = _align(len(image), file_alignment)
    raw_size = _align(len(section_data), file_alignment)
    image.extend(b"\0" * (raw_offset - len(image)))
    image.extend(section_data)
    image.extend(b"\0" * (raw_size - len(section_data)))

    section_header = struct.pack(
        "<8sIIIIIIHHI",
        b".p2rsrc\0",
        len(section_data),
        section_rva,
        raw_size,
        raw_offset,
        0,
        0,
        0,
        0,
        0x40000040,
    )
    image[header_offset : header_offset + 40] = section_header
    coff = int(layout["coff"])
    optional = int(layout["optional"])
    struct.pack_into("<H", image, coff + 2, section_count + 1)
    struct.pack_into(
        "<I",
        image,
        optional + 56,
        _align(section_rva + len(section_data), section_alignment),
    )
    struct.pack_into("<II", image, optional + 112 + 16, section_rva, len(section_data))
    struct.pack_into("<I", image, optional + 64, 0)  # checksum
    struct.pack_into("<II", image, optional + 112 + 32, 0, 0)  # certificate table
    executable.write_bytes(image)
