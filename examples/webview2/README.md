# WebView2, with nothing but py2bin

**The vendor's `WebView2.h` compiles now.** All 68,921 lines of it, straight
out of the NuGet package, for both Windows targets:

```sh
py2bin fetch-header WebView2.h --into vendor
py2bin cc app.c -o app.exe --target windows-x86_64 --include-dir vendor
```

It reads because MIDL declares every interface twice — C++ classes in one
branch, a table of function pointers in the other — and py2bin defines no
`__cplusplus`, so the table is the branch it gets. Which is the branch it
wants: a COM object *is* that table, and py2bin's C compiles one. The headers
that branch reaches for — `unknwn.h`, `objidl.h`, `oaidl.h`, `EventToken.h`,
`sal.h`, `rpc.h` — are py2bin's own, for the same reason as always: nobody
publishes them as files.

So a program written against the vendor's header calls the vendor's own
slots. `Navigate` is slot 5 in `ICoreWebView2Vtbl`, and what comes out loads
offset 0x28.

`vendor_header.c` here is written that way: it calls
`ICoreWebView2Controller::get_CoreWebView2` and `ICoreWebView2::Navigate`
through the vendor's own tables. It needs the header, which is a download
rather than a file in this repository:

```sh
py2bin fetch-header WebView2.h --into vendor
py2bin cc vendor_header.c -o app.exe --target windows-x86_64 --include-dir vendor
```

The rest of this file describes the route taken before that worked, which is
still the smaller one and still builds. What a caller needs from a COM
interface is less than the whole header: an interface is a table of function
pointers, and a call is a load from a fixed slot. So the slots can be read out
of the vendor's header and written as a class py2bin compiles — which is what
`webview2_min.h` here is.

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
