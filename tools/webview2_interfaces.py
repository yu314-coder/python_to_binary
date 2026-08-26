#!/usr/bin/env python3
"""Write a COM interface header py2bin can compile, from one it cannot.

The vendor's `WebView2.h` is MIDL output: it includes the COM headers, and
every open implementation of those generates them from `.idl` at build time,
so there is no file to fetch and nothing to compile against. But the header
can still be *read*, and what a caller needs from an interface is only the
order of its methods - a call through one is a load from a table at a fixed
slot.

So this reads the slots out of the vendor's header, in its own order, and
writes them as a class in py2bin's C++ subset. A method a program calls gets
its real signature; the rest hold their place in the table.

    python3 tools/webview2_interfaces.py path/to/WebView2.h > webview2_min.h

`py2bin fetch-header WebView2.h --into vendor` gets the input.
"""

import re
import sys
from pathlib import Path

#: `MIDL_INTERFACE("...")\nNAME : public BASE {`
_DECLARED = r'MIDL_INTERFACE\("[0-9A-Fa-f-]+"\)\s*\n\s*{name}\s*:\s*public\s+\w+'
#: One method. The annotations MIDL writes between `virtual` and the result
#: are comments, and a property's getter is a slot like any other.
_SLOT = re.compile(
    r"virtual\s+(?:/\*.*?\*/\s*)*HRESULT\s+STDMETHODCALLTYPE\s+(\w+)\s*\("
)

#: The ones a program actually calls, with the signature each really has.
#: Everything else is a placeholder: it exists so the slots after it are at
#: the index the vendor put them at.
SPELLED = {
    "CreateCoreWebView2Controller":
        "HWND parent, ICoreWebView2CreateCoreWebView2ControllerCompletedHandler *handler",
    "get_CoreWebView2": "ICoreWebView2 **answer",
    "put_Bounds": "RECT bounds",
    "put_IsVisible": "BOOL visible",
    "Navigate": "LPCWSTR uri",
    "NavigateToString": "LPCWSTR html",
    "ExecuteScript":
        "LPCWSTR script, ICoreWebView2ExecuteScriptCompletedHandler *handler",
    "PostWebMessageAsString": "LPCWSTR message",
    "Reload": "",
    "Stop": "",
    "GoBack": "",
    "GoForward": "",
}

WANTED = (
    "ICoreWebView2Environment",
    "ICoreWebView2Controller",
    "ICoreWebView2",
)


def body_of(text: str, name: str) -> str:
    """The braces of one interface declaration."""

    at = re.search(_DECLARED.format(name=name), text)
    if at is None:
        raise SystemExit(f"{name} is not declared in that header")
    start = text.index("{", at.end())
    depth = 0
    index = start
    while index < len(text):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return text[start:index]
        index += 1
    raise SystemExit(f"{name} is not closed")


def main(argv: "list[str]") -> int:
    if len(argv) != 2:
        print(__doc__.strip(), file=sys.stderr)
        return 2
    text = Path(argv[1]).read_text(errors="replace")
    out = [_HEAD]
    for name in WANTED:
        slots = _SLOT.findall(body_of(text, name))
        out.append(f"\n/* {len(slots)} slots, in the vendor header's order. */")
        out.append(f"class {name} : public IUnknown {{\npublic:")
        for index, slot in enumerate(slots):
            if slot in SPELLED:
                out.append(f"    virtual HRESULT {slot}({SPELLED[slot]}) = 0;")
            else:
                out.append(f"    virtual HRESULT {slot}(void *unused_{index}) = 0;")
        out.append("};")
    out.append(_TAIL)
    print("\n".join(out))
    return 0


_HEAD = '''/* WebView2's interfaces, in py2bin's C++ subset.
 *
 * Written by tools/webview2_interfaces.py from the vendor's own header, so
 * every slot is at the index that header put it at and a call lands where it
 * says. The ones a program usually calls carry their real signature; the rest
 * hold their place in the table, and calling one would be a mistake this
 * cannot catch.
 */
#ifndef PY2BIN_WEBVIEW2_H
#define PY2BIN_WEBVIEW2_H
#include <unknwn.h>

typedef void *HWND;
typedef int BOOL;
typedef const wchar_t *LPCWSTR;
typedef struct __webview2_rect { long left; long top; long right; long bottom; } RECT;

class ICoreWebView2;
class ICoreWebView2Controller;
class ICoreWebView2Environment;
class ICoreWebView2CreateCoreWebView2ControllerCompletedHandler;
class ICoreWebView2ExecuteScriptCompletedHandler;
'''

_TAIL = '''
/* The entry point the loader binds. Everything else is reached through the
   interfaces above. */
HRESULT CreateCoreWebView2EnvironmentWithOptions(
    LPCWSTR browserExecutableFolder, LPCWSTR userDataFolder,
    void *environmentOptions, void *environmentCreatedHandler);

#endif'''


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
