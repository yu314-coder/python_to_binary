#include <windows.h>
#include <WebView2.h>

/* The loader entry point, reached by name so the vendor's DLL is not in the
   import table. */
typedef HRESULT (*PFN_CREATE_ENV)(PCWSTR, PCWSTR, void *, void *);

static ICoreWebView2Controller *g_controller = 0;

static int navigate(ICoreWebView2Controller *controller, LPCWSTR url) {
    ICoreWebView2 *view = 0;
    HRESULT hr = controller->lpVtbl->get_CoreWebView2(controller, &view);
    if (FAILED(hr) || view == 0) { return 1; }
    hr = view->lpVtbl->Navigate(view, (LPCWSTR)url);
    view->lpVtbl->Release(view);
    return SUCCEEDED(hr) ? 0 : 2;
}

int main(void) {
    CoInitializeEx((LPVOID)0, COINIT_APARTMENTTHREADED);
    HMODULE loader = LoadLibraryW(L"WebView2Loader.dll");
    if (loader == (HMODULE)0) { CoUninitialize(); return 3; }
    PFN_CREATE_ENV create = (PFN_CREATE_ENV)GetProcAddress(
        loader, "CreateCoreWebView2EnvironmentWithOptions");
    if (create == (PFN_CREATE_ENV)0) { CoUninitialize(); return 4; }
    HRESULT hr = create((PCWSTR)0, (PCWSTR)0, (void *)0, (void *)0);
    int rc = g_controller ? navigate(g_controller, L"https://example.com") : 0;
    FreeLibrary(loader);
    CoUninitialize();
    return SUCCEEDED(hr) ? rc : 5;
}
