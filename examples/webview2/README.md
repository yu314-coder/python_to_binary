# WebView2, with nothing but py2bin

The vendor's `WebView2.h` cannot be compiled here, and no amount of fetching
changes that. It is MIDL output: it includes `objbase.h`, `oaidl.h`,
`unknwn.h` and the rest, and **every** open implementation of those generates
them from `.idl` with a tool that runs at build time. Checked against three of
them — none publishes one as a file. The vendor's own set ships inside a
toolchain.

What a caller needs from a COM interface is smaller than the header: an
interface is a table of function pointers, and a call is a load from a fixed
slot. So the answer is to read the slots out of the vendor's header — which
fetches fine, it just cannot be compiled — and write them as a class py2bin
does compile.

```sh
py2bin fetch-header WebView2.h --into vendor
python3 tools/webview2_interfaces.py vendor/WebView2.h > webview2_min.h
python3 build.py main.cpp --target windows-x86_64
```

`webview2_min.h` here was written that way. Every slot is at the index the
vendor put it at, so a call lands where it says. The methods a program usually
calls carry their real signature; the rest hold their place in the table, and
calling one of those is a mistake nothing here can catch — add its signature
to `SPELLED` in the generator when you need it.

`main.cpp` calls through the interfaces the way a real program does. It stands
its own object behind them rather than the one the loader hands back, so the
example runs on any target: what is being shown is the dispatch, and that part
is the same either way. To talk to the real WebView2, keep the calls and let
`CreateCoreWebView2EnvironmentWithOptions` — declared at the end of the
generated header, bound by the loader like any other import — give you the
environment.

`<unknwn.h>` is py2bin's own, for the same reason this file exists.
