from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class LibraryHook:
    module: str
    distributions: tuple[str, ...]
    note: str


HOOKS = {
    hook.module: hook
    for hook in (
        LibraryHook(
            "torch",
            ("torch",),
            "PyTorch native libraries are collected; build on the same OS, CPU "
            "architecture, Python ABI, and accelerator family as the target.",
        ),
        LibraryHook(
            "transformers",
            ("transformers",),
            "Transformers code is collected. Model weights are runtime data: place "
            "them in the project or pre-populate a target-side cache for offline use.",
        ),
        LibraryHook(
            "manim",
            ("manim",),
            "Manim Python packages are collected. External programs such as ffmpeg, "
            "LaTeX, and system fonts must be installed or shipped separately.",
        ),
        LibraryHook(
            "bpy",
            ("bpy",),
            "bpy is tied to a Blender/Python version. Build with Blender's Python or "
            "a compatible bpy wheel; Blender resources may need explicit inclusion.",
        ),
        LibraryHook(
            "numpy",
            ("numpy",),
            "NumPy extension modules are collected and require a compatible target ABI.",
        ),
    )
}


def hooks_for(modules: set[str]) -> list[LibraryHook]:
    roots = {module.partition(".")[0] for module in modules}
    return [HOOKS[name] for name in sorted(roots & HOOKS.keys())]

