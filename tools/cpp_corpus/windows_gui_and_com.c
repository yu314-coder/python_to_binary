/* Windows only: builds for the two Windows targets and is skipped elsewhere,
   which the sweep does by reading the guard below. What it covers is the
   twelve-argument CreateWindowExW - more arguments than there are registers -
   and coming by a COM pointer, which needs ole32 and a vendor DLL loaded by
   name rather than named in the import table. */
#include <stdio.h>

#ifdef _WIN32
#include <windows.h>

typedef HRESULT (*PFN_CREATE_ENV)(LPCWSTR, LPCWSTR, LPVOID, LPVOID);

static LRESULT CALLBACK WndProc(HWND hwnd, UINT msg, WPARAM wp, LPARAM lp) {
    if (msg == WM_DESTROY) { PostQuitMessage(0); return (LRESULT)0; }
    return DefWindowProcW(hwnd, msg, wp, lp);
}

int main(void) {
    HINSTANCE me = GetModuleHandleW((LPCWSTR)0);
    WNDCLASSEXW cls;
    cls.cbSize = sizeof(WNDCLASSEXW);
    cls.style = 0;
    cls.lpfnWndProc = WndProc;
    cls.cbClsExtra = 0;
    cls.cbWndExtra = 0;
    cls.hInstance = me;
    cls.hIcon = LoadIconW((HINSTANCE)0, IDI_APPLICATION);
    cls.hCursor = LoadCursorW((HINSTANCE)0, IDC_ARROW);
    cls.hbrBackground = (HBRUSH)(COLOR_WINDOW + 1);
    cls.lpszMenuName = (LPCWSTR)0;
    cls.lpszClassName = L"py2binWindow";
    cls.hIconSm = cls.hIcon;
    RegisterClassExW(&cls);
    HWND w = CreateWindowExW(0, L"py2binWindow", L"py2bin",
                             WS_OVERLAPPEDWINDOW,
                             CW_USEDEFAULT, CW_USEDEFAULT, 900, 600,
                             (HWND)0, (HMENU)0, me, (LPVOID)0);
    ShowWindow(w, SW_SHOW);
    UpdateWindow(w);
    SetWindowPos(w, (HWND)0, 0, 0, 800, 500, 0);

    if (SUCCEEDED(CoInitializeEx((LPVOID)0, COINIT_APARTMENTTHREADED))) {
        BSTR text = SysAllocString(L"hello");
        UINT n = SysStringLen(text);
        SysFreeString(text);
        HMODULE loader = LoadLibraryW(L"WebView2Loader.dll");
        if (loader != (HMODULE)0) {
            PFN_CREATE_ENV create = (PFN_CREATE_ENV)GetProcAddress(
                loader, "CreateCoreWebView2EnvironmentWithOptions");
            if (create != (PFN_CREATE_ENV)0) { create(L"", L"", (LPVOID)0, (LPVOID)0); }
            FreeLibrary(loader);
        }
        CoUninitialize();
        printf("%u\n", n);
    }
    DestroyWindow(w);
    UnregisterClassW(L"py2binWindow", me);
    return 0;
}
#else
int main(void) { printf("5\n"); return 0; }
#endif
