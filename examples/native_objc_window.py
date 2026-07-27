"""A real macOS window showing HTML in a WKWebView, with no Python at run time.

Compiled for darwin-arm64 this is a Mach-O binary that contains no CPython at
all. It reaches Cocoa the way a C program does -- through the Objective-C
runtime's objc_getClass/sel_registerName/objc_msgSend -- and it builds its own
delegate class at run time so that AppKit and WebKit can call back into
compiled code.

The three methods below are ordinary Python functions. Each is compiled into a
function with a real stack frame whose address is handed to class_addMethod, so
the framework branches straight into this image with the receiver in x0 and the
selector in x1. Nothing here polls or fakes an event: the window closes because
WebKit finished loading the page and told us so.

The program waits for you to close the window; that is what ends it. An
earlier version closed the window itself as soon as WebKit reported the page
loaded, which is before WebKit has drawn anything - a real window, a real web
view, and nothing on screen.

Run it either way and it does the same thing::

    PYTHONPATH=src python3 -u examples/native_objc_window.py
    PYTHONPATH=src python3 -m py2bin compile examples/native_objc_window.py \
        --target darwin-arm64 -o window.bin && ./window.bin

The ``-u`` is not decoration and it is not py2bin's doing. AppKit ends this
program by calling C ``exit()`` from inside -[NSApplication terminate:], which
never reaches the flush CPython does on its way out, so anything still sitting
in Python's stdout buffer is discarded. Compiled, every print is a write(2)
that has already happened. Unbuffered stdout is what makes the two runs
comparable; with a pipe and no ``-u``, CPython prints nothing at all.
"""

from py2bin.cabi import (
    objc_getClass,
    sel_registerName,
    objc_msgSend,
    objc_msgSend2,
    objc_msgSend_str,
    objc_msgSend_id_id,
    objc_msgSend_long,
    objc_msgSend_bool_void,
    objc_msgSend_rect,
    objc_msgSend_rect_id,
    objc_msgSend_rect_uint_uint_bool,
    objc_allocateClassPair,
    class_addMethod,
    objc_registerClassPair,
)

PAGE = (
    "<html><body style='font:32px -apple-system;margin:60px'>"
    "<h1>py2bin</h1><p>native Cocoa, no CPython</p></body></html>"
)


def application_did_finish_launching(this, cmd, notification):
    """-[NSApplicationDelegate applicationDidFinishLaunching:], encoding v@:@."""

    print("applicationDidFinishLaunching: ran in native code")


def should_terminate_after_last_window(this, cmd, sender):
    """-applicationShouldTerminateAfterLastWindowClosed:, encoding B@:@."""

    print("applicationShouldTerminateAfterLastWindowClosed: ran in native code")
    return 1


def web_view_did_finish(this, cmd, web_view, navigation):
    """-[WKNavigationDelegate webView:didFinishNavigation:], encoding v@:@@.

    WebKit calls this once the page is up, so it is proof that the view really
    rendered rather than merely existing. Closing the window from here is what
    lets the program end on its own; the receiver's own ``window`` is reachable
    through the web view that WebKit hands in, which matters because a method
    implementation cannot read the module's variables.
    """

    print("webView:didFinishNavigation: ran in native code")
    # Do NOT close the window here. This fires when WebKit finishes loading,
    # which is before the page has been drawn: closing now leaves a real window
    # that was never painted, which looks exactly like a failure. The window
    # stays until it is closed, and closing it ends the program through
    # applicationShouldTerminateAfterLastWindowClosed: below.
    print("the page is loaded; close the window to exit")


NSApplication = objc_getClass("NSApplication")
NSWindow = objc_getClass("NSWindow")
NSString = objc_getClass("NSString")
WKWebView = objc_getClass("WKWebView")
WKWebViewConfiguration = objc_getClass("WKWebViewConfiguration")

alloc = sel_registerName("alloc")
init = sel_registerName("init")
new = sel_registerName("new")
with_utf8 = sel_registerName("stringWithUTF8String:")

# Build the delegate class. It answers for the application and for the web
# view's navigation, which is two informal protocols on one object -- exactly
# what a small Cocoa program does. class_addMethod must come before
# objc_registerClassPair: after registration the class is published and the
# runtime wants class_replaceMethod instead.
Delegate = objc_allocateClassPair(objc_getClass("NSObject"), "Py2BinDelegate", 0)
class_addMethod(
    Delegate,
    sel_registerName("applicationDidFinishLaunching:"),
    application_did_finish_launching,
    "v@:@",
)
class_addMethod(
    Delegate,
    sel_registerName("applicationShouldTerminateAfterLastWindowClosed:"),
    should_terminate_after_last_window,
    "B@:@",
)
class_addMethod(
    Delegate,
    sel_registerName("webView:didFinishNavigation:"),
    web_view_did_finish,
    "v@:@@",
)
objc_registerClassPair(Delegate)

app = objc_msgSend(NSApplication, sel_registerName("sharedApplication"))
# NSApplicationActivationPolicyRegular. Without it a bare executable has no
# place in the Dock and its windows never come forward.
objc_msgSend_long(app, sel_registerName("setActivationPolicy:"), 0)
delegate = objc_msgSend(Delegate, new)
objc_msgSend2(app, sel_registerName("setDelegate:"), delegate)

# NSWindowStyleMaskTitled|Closable|Miniaturizable|Resizable, and
# NSBackingStoreBuffered.
window = objc_msgSend_rect_uint_uint_bool(
    objc_msgSend(NSWindow, alloc),
    sel_registerName("initWithContentRect:styleMask:backing:defer:"),
    120.0, 120.0, 640.0, 420.0,
    15, 2, 0,
)
objc_msgSend2(
    window,
    sel_registerName("setTitle:"),
    objc_msgSend_str(NSString, with_utf8, "py2bin native window"),
)

configuration = objc_msgSend(objc_msgSend(WKWebViewConfiguration, alloc), init)
web_view = objc_msgSend_rect_id(
    objc_msgSend(WKWebView, alloc),
    sel_registerName("initWithFrame:configuration:"),
    0.0, 0.0, 640.0, 420.0,
    configuration,
)
objc_msgSend2(web_view, sel_registerName("setNavigationDelegate:"), delegate)
objc_msgSend_id_id(
    web_view,
    sel_registerName("loadHTMLString:baseURL:"),
    objc_msgSend_str(NSString, with_utf8, PAGE),
    0,
)
objc_msgSend2(window, sel_registerName("setContentView:"), web_view)
objc_msgSend2(window, sel_registerName("makeKeyAndOrderFront:"), 0)
objc_msgSend_bool_void(app, sel_registerName("activateIgnoringOtherApps:"), 1)

print("entering the AppKit run loop")
objc_msgSend(app, sel_registerName("run"))
