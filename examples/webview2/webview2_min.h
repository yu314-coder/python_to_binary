/* WebView2's interfaces, in py2bin's C++ subset.
 *
 * Written by tools/webview2_interfaces.py from the vendor's own header, so
 * every slot is at the index that header put it at, in the convention it
 * declares it in, and a call lands where it says. The ones a program usually
 * calls carry their real signature; the rest hold their place in the table,
 * and calling one would be a mistake this cannot catch.
 */
#ifndef PY2BIN_WEBVIEW2_H
#define PY2BIN_WEBVIEW2_H
#ifdef _WIN32
#include <windows.h>
#endif
#include <unknwn.h>

#ifndef _WIN32
/* The vendor's header takes these from <windows.h>, which is Windows' own and
   refused anywhere else. <unknwn.h> brings BOOL and LPCWSTR on every target;
   these are the rest of what the signatures below name, so the example builds
   anywhere. */
typedef void *HWND;
typedef struct tagRECT { long left; long top; long right; long bottom; } RECT;
#endif

class ICoreWebView2;
class ICoreWebView2Controller;
class ICoreWebView2Environment;
class ICoreWebView2CreateCoreWebView2ControllerCompletedHandler;
class ICoreWebView2ExecuteScriptCompletedHandler;


/* 5 slots, in the vendor header's order. */
class ICoreWebView2Environment : public IUnknown {
public:
    virtual HRESULT STDMETHODCALLTYPE CreateCoreWebView2Controller(HWND parent, ICoreWebView2CreateCoreWebView2ControllerCompletedHandler *handler) = 0;
    virtual HRESULT STDMETHODCALLTYPE CreateWebResourceResponse(void *unused_1) = 0;
    virtual HRESULT STDMETHODCALLTYPE get_BrowserVersionString(void *unused_2) = 0;
    virtual HRESULT STDMETHODCALLTYPE add_NewBrowserVersionAvailable(void *unused_3) = 0;
    virtual HRESULT STDMETHODCALLTYPE remove_NewBrowserVersionAvailable(void *unused_4) = 0;
};

/* 23 slots, in the vendor header's order. */
class ICoreWebView2Controller : public IUnknown {
public:
    virtual HRESULT STDMETHODCALLTYPE get_IsVisible(void *unused_0) = 0;
    virtual HRESULT STDMETHODCALLTYPE put_IsVisible(BOOL visible) = 0;
    virtual HRESULT STDMETHODCALLTYPE get_Bounds(void *unused_2) = 0;
    virtual HRESULT STDMETHODCALLTYPE put_Bounds(RECT bounds) = 0;
    virtual HRESULT STDMETHODCALLTYPE get_ZoomFactor(void *unused_4) = 0;
    virtual HRESULT STDMETHODCALLTYPE put_ZoomFactor(void *unused_5) = 0;
    virtual HRESULT STDMETHODCALLTYPE add_ZoomFactorChanged(void *unused_6) = 0;
    virtual HRESULT STDMETHODCALLTYPE remove_ZoomFactorChanged(void *unused_7) = 0;
    virtual HRESULT STDMETHODCALLTYPE SetBoundsAndZoomFactor(void *unused_8) = 0;
    virtual HRESULT STDMETHODCALLTYPE MoveFocus(void *unused_9) = 0;
    virtual HRESULT STDMETHODCALLTYPE add_MoveFocusRequested(void *unused_10) = 0;
    virtual HRESULT STDMETHODCALLTYPE remove_MoveFocusRequested(void *unused_11) = 0;
    virtual HRESULT STDMETHODCALLTYPE add_GotFocus(void *unused_12) = 0;
    virtual HRESULT STDMETHODCALLTYPE remove_GotFocus(void *unused_13) = 0;
    virtual HRESULT STDMETHODCALLTYPE add_LostFocus(void *unused_14) = 0;
    virtual HRESULT STDMETHODCALLTYPE remove_LostFocus(void *unused_15) = 0;
    virtual HRESULT STDMETHODCALLTYPE add_AcceleratorKeyPressed(void *unused_16) = 0;
    virtual HRESULT STDMETHODCALLTYPE remove_AcceleratorKeyPressed(void *unused_17) = 0;
    virtual HRESULT STDMETHODCALLTYPE get_ParentWindow(void *unused_18) = 0;
    virtual HRESULT STDMETHODCALLTYPE put_ParentWindow(void *unused_19) = 0;
    virtual HRESULT STDMETHODCALLTYPE NotifyParentWindowPositionChanged(void *unused_20) = 0;
    virtual HRESULT STDMETHODCALLTYPE Close(void *unused_21) = 0;
    virtual HRESULT STDMETHODCALLTYPE get_CoreWebView2(ICoreWebView2 **answer) = 0;
};

/* 58 slots, in the vendor header's order. */
class ICoreWebView2 : public IUnknown {
public:
    virtual HRESULT STDMETHODCALLTYPE get_Settings(void *unused_0) = 0;
    virtual HRESULT STDMETHODCALLTYPE get_Source(void *unused_1) = 0;
    virtual HRESULT STDMETHODCALLTYPE Navigate(LPCWSTR uri) = 0;
    virtual HRESULT STDMETHODCALLTYPE NavigateToString(LPCWSTR html) = 0;
    virtual HRESULT STDMETHODCALLTYPE add_NavigationStarting(void *unused_4) = 0;
    virtual HRESULT STDMETHODCALLTYPE remove_NavigationStarting(void *unused_5) = 0;
    virtual HRESULT STDMETHODCALLTYPE add_ContentLoading(void *unused_6) = 0;
    virtual HRESULT STDMETHODCALLTYPE remove_ContentLoading(void *unused_7) = 0;
    virtual HRESULT STDMETHODCALLTYPE add_SourceChanged(void *unused_8) = 0;
    virtual HRESULT STDMETHODCALLTYPE remove_SourceChanged(void *unused_9) = 0;
    virtual HRESULT STDMETHODCALLTYPE add_HistoryChanged(void *unused_10) = 0;
    virtual HRESULT STDMETHODCALLTYPE remove_HistoryChanged(void *unused_11) = 0;
    virtual HRESULT STDMETHODCALLTYPE add_NavigationCompleted(void *unused_12) = 0;
    virtual HRESULT STDMETHODCALLTYPE remove_NavigationCompleted(void *unused_13) = 0;
    virtual HRESULT STDMETHODCALLTYPE add_FrameNavigationStarting(void *unused_14) = 0;
    virtual HRESULT STDMETHODCALLTYPE remove_FrameNavigationStarting(void *unused_15) = 0;
    virtual HRESULT STDMETHODCALLTYPE add_FrameNavigationCompleted(void *unused_16) = 0;
    virtual HRESULT STDMETHODCALLTYPE remove_FrameNavigationCompleted(void *unused_17) = 0;
    virtual HRESULT STDMETHODCALLTYPE add_ScriptDialogOpening(void *unused_18) = 0;
    virtual HRESULT STDMETHODCALLTYPE remove_ScriptDialogOpening(void *unused_19) = 0;
    virtual HRESULT STDMETHODCALLTYPE add_PermissionRequested(void *unused_20) = 0;
    virtual HRESULT STDMETHODCALLTYPE remove_PermissionRequested(void *unused_21) = 0;
    virtual HRESULT STDMETHODCALLTYPE add_ProcessFailed(void *unused_22) = 0;
    virtual HRESULT STDMETHODCALLTYPE remove_ProcessFailed(void *unused_23) = 0;
    virtual HRESULT STDMETHODCALLTYPE AddScriptToExecuteOnDocumentCreated(void *unused_24) = 0;
    virtual HRESULT STDMETHODCALLTYPE RemoveScriptToExecuteOnDocumentCreated(void *unused_25) = 0;
    virtual HRESULT STDMETHODCALLTYPE ExecuteScript(LPCWSTR script, ICoreWebView2ExecuteScriptCompletedHandler *handler) = 0;
    virtual HRESULT STDMETHODCALLTYPE CapturePreview(void *unused_27) = 0;
    virtual HRESULT STDMETHODCALLTYPE Reload() = 0;
    virtual HRESULT STDMETHODCALLTYPE PostWebMessageAsJson(void *unused_29) = 0;
    virtual HRESULT STDMETHODCALLTYPE PostWebMessageAsString(LPCWSTR message) = 0;
    virtual HRESULT STDMETHODCALLTYPE add_WebMessageReceived(void *unused_31) = 0;
    virtual HRESULT STDMETHODCALLTYPE remove_WebMessageReceived(void *unused_32) = 0;
    virtual HRESULT STDMETHODCALLTYPE CallDevToolsProtocolMethod(void *unused_33) = 0;
    virtual HRESULT STDMETHODCALLTYPE get_BrowserProcessId(void *unused_34) = 0;
    virtual HRESULT STDMETHODCALLTYPE get_CanGoBack(void *unused_35) = 0;
    virtual HRESULT STDMETHODCALLTYPE get_CanGoForward(void *unused_36) = 0;
    virtual HRESULT STDMETHODCALLTYPE GoBack() = 0;
    virtual HRESULT STDMETHODCALLTYPE GoForward() = 0;
    virtual HRESULT STDMETHODCALLTYPE GetDevToolsProtocolEventReceiver(void *unused_39) = 0;
    virtual HRESULT STDMETHODCALLTYPE Stop() = 0;
    virtual HRESULT STDMETHODCALLTYPE add_NewWindowRequested(void *unused_41) = 0;
    virtual HRESULT STDMETHODCALLTYPE remove_NewWindowRequested(void *unused_42) = 0;
    virtual HRESULT STDMETHODCALLTYPE add_DocumentTitleChanged(void *unused_43) = 0;
    virtual HRESULT STDMETHODCALLTYPE remove_DocumentTitleChanged(void *unused_44) = 0;
    virtual HRESULT STDMETHODCALLTYPE get_DocumentTitle(void *unused_45) = 0;
    virtual HRESULT STDMETHODCALLTYPE AddHostObjectToScript(void *unused_46) = 0;
    virtual HRESULT STDMETHODCALLTYPE RemoveHostObjectFromScript(void *unused_47) = 0;
    virtual HRESULT STDMETHODCALLTYPE OpenDevToolsWindow(void *unused_48) = 0;
    virtual HRESULT STDMETHODCALLTYPE add_ContainsFullScreenElementChanged(void *unused_49) = 0;
    virtual HRESULT STDMETHODCALLTYPE remove_ContainsFullScreenElementChanged(void *unused_50) = 0;
    virtual HRESULT STDMETHODCALLTYPE get_ContainsFullScreenElement(void *unused_51) = 0;
    virtual HRESULT STDMETHODCALLTYPE add_WebResourceRequested(void *unused_52) = 0;
    virtual HRESULT STDMETHODCALLTYPE remove_WebResourceRequested(void *unused_53) = 0;
    virtual HRESULT STDMETHODCALLTYPE AddWebResourceRequestedFilter(void *unused_54) = 0;
    virtual HRESULT STDMETHODCALLTYPE RemoveWebResourceRequestedFilter(void *unused_55) = 0;
    virtual HRESULT STDMETHODCALLTYPE add_WindowCloseRequested(void *unused_56) = 0;
    virtual HRESULT STDMETHODCALLTYPE remove_WindowCloseRequested(void *unused_57) = 0;
};

/* The entry point the loader binds. Everything else is reached through the
   interfaces above. */
HRESULT CreateCoreWebView2EnvironmentWithOptions(
    LPCWSTR browserExecutableFolder, LPCWSTR userDataFolder,
    void *environmentOptions, void *environmentCreatedHandler);

#endif
