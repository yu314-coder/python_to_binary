/* A WebView2 program, built with nothing but py2bin.
 *
 * It does not include the vendor's WebView2.h - that header is MIDL output
 * and reaches for COM headers no implementation publishes as files. It
 * includes the interface declarations beside it instead, whose slot order was
 * read out of the vendor header itself.
 */
#include <stdio.h>
#include "webview2_min.h"

/* What a program does with the interfaces: hold one, and call through it.
   Every call here goes through the object's own table at the slot the
   vendor's header put it at. */
static HRESULT show(ICoreWebView2Controller *controller, LPCWSTR uri) {
    RECT bounds;
    bounds.left = 0; bounds.top = 0; bounds.right = 900; bounds.bottom = 600;
    HRESULT hr = controller->put_Bounds(bounds);
    if (FAILED(hr)) { return hr; }
    hr = controller->put_IsVisible(1);
    if (FAILED(hr)) { return hr; }
    ICoreWebView2 *view = 0;
    hr = controller->get_CoreWebView2(&view);
    if (FAILED(hr)) { return hr; }
    return view->Navigate(uri);
}

/* Standing in for the real one, so the example runs anywhere: the same calls,
   against an object this file implements rather than one the loader hands
   back. What is being demonstrated is the dispatch, and that is the same. */
class FakeView : public ICoreWebView2 {
public:
    unsigned long refs;
    const wchar_t *went;
    FakeView() { refs = 1; went = 0; }
    HRESULT QueryInterface(REFIID riid, void **object) { *object = this; return S_OK; }
    unsigned long AddRef() { refs = refs + 1; return refs; }
    unsigned long Release() { refs = refs - 1; return refs; }
    HRESULT Navigate(LPCWSTR uri) { went = uri; return S_OK; }
};

int main() {
    FakeView view;
    ICoreWebView2 *held = &view;
    HRESULT hr = held->Navigate(L"https://example.com");
    printf("%d %d %d\n", (int)SUCCEEDED(hr), (int)(view.went != 0),
           (int)held->AddRef());
    return 0;
}
