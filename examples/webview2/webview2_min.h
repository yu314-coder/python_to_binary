/* WebView2's interfaces, in py2bin's C++ subset.
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


/* 5 slots, in the vendor header's order. */
class ICoreWebView2Environment : public IUnknown {
public:
    virtual HRESULT CreateCoreWebView2Controller(HWND parent, ICoreWebView2CreateCoreWebView2ControllerCompletedHandler *handler) = 0;
    virtual HRESULT CreateWebResourceResponse(void *unused_1) = 0;
    virtual HRESULT get_BrowserVersionString(void *unused_2) = 0;
    virtual HRESULT add_NewBrowserVersionAvailable(void *unused_3) = 0;
    virtual HRESULT remove_NewBrowserVersionAvailable(void *unused_4) = 0;
};

/* 23 slots, in the vendor header's order. */
class ICoreWebView2Controller : public IUnknown {
public:
    virtual HRESULT get_IsVisible(void *unused_0) = 0;
    virtual HRESULT put_IsVisible(BOOL visible) = 0;
    virtual HRESULT get_Bounds(void *unused_2) = 0;
    virtual HRESULT put_Bounds(RECT bounds) = 0;
    virtual HRESULT get_ZoomFactor(void *unused_4) = 0;
    virtual HRESULT put_ZoomFactor(void *unused_5) = 0;
    virtual HRESULT add_ZoomFactorChanged(void *unused_6) = 0;
    virtual HRESULT remove_ZoomFactorChanged(void *unused_7) = 0;
    virtual HRESULT SetBoundsAndZoomFactor(void *unused_8) = 0;
    virtual HRESULT MoveFocus(void *unused_9) = 0;
    virtual HRESULT add_MoveFocusRequested(void *unused_10) = 0;
    virtual HRESULT remove_MoveFocusRequested(void *unused_11) = 0;
    virtual HRESULT add_GotFocus(void *unused_12) = 0;
    virtual HRESULT remove_GotFocus(void *unused_13) = 0;
    virtual HRESULT add_LostFocus(void *unused_14) = 0;
    virtual HRESULT remove_LostFocus(void *unused_15) = 0;
    virtual HRESULT add_AcceleratorKeyPressed(void *unused_16) = 0;
    virtual HRESULT remove_AcceleratorKeyPressed(void *unused_17) = 0;
    virtual HRESULT get_ParentWindow(void *unused_18) = 0;
    virtual HRESULT put_ParentWindow(void *unused_19) = 0;
    virtual HRESULT NotifyParentWindowPositionChanged(void *unused_20) = 0;
    virtual HRESULT Close(void *unused_21) = 0;
    virtual HRESULT get_CoreWebView2(ICoreWebView2 **answer) = 0;
};

/* 58 slots, in the vendor header's order. */
class ICoreWebView2 : public IUnknown {
public:
    virtual HRESULT get_Settings(void *unused_0) = 0;
    virtual HRESULT get_Source(void *unused_1) = 0;
    virtual HRESULT Navigate(LPCWSTR uri) = 0;
    virtual HRESULT NavigateToString(LPCWSTR html) = 0;
    virtual HRESULT add_NavigationStarting(void *unused_4) = 0;
    virtual HRESULT remove_NavigationStarting(void *unused_5) = 0;
    virtual HRESULT add_ContentLoading(void *unused_6) = 0;
    virtual HRESULT remove_ContentLoading(void *unused_7) = 0;
    virtual HRESULT add_SourceChanged(void *unused_8) = 0;
    virtual HRESULT remove_SourceChanged(void *unused_9) = 0;
    virtual HRESULT add_HistoryChanged(void *unused_10) = 0;
    virtual HRESULT remove_HistoryChanged(void *unused_11) = 0;
    virtual HRESULT add_NavigationCompleted(void *unused_12) = 0;
    virtual HRESULT remove_NavigationCompleted(void *unused_13) = 0;
    virtual HRESULT add_FrameNavigationStarting(void *unused_14) = 0;
    virtual HRESULT remove_FrameNavigationStarting(void *unused_15) = 0;
    virtual HRESULT add_FrameNavigationCompleted(void *unused_16) = 0;
    virtual HRESULT remove_FrameNavigationCompleted(void *unused_17) = 0;
    virtual HRESULT add_ScriptDialogOpening(void *unused_18) = 0;
    virtual HRESULT remove_ScriptDialogOpening(void *unused_19) = 0;
    virtual HRESULT add_PermissionRequested(void *unused_20) = 0;
    virtual HRESULT remove_PermissionRequested(void *unused_21) = 0;
    virtual HRESULT add_ProcessFailed(void *unused_22) = 0;
    virtual HRESULT remove_ProcessFailed(void *unused_23) = 0;
    virtual HRESULT AddScriptToExecuteOnDocumentCreated(void *unused_24) = 0;
    virtual HRESULT RemoveScriptToExecuteOnDocumentCreated(void *unused_25) = 0;
    virtual HRESULT ExecuteScript(LPCWSTR script, ICoreWebView2ExecuteScriptCompletedHandler *handler) = 0;
    virtual HRESULT CapturePreview(void *unused_27) = 0;
    virtual HRESULT Reload() = 0;
    virtual HRESULT PostWebMessageAsJson(void *unused_29) = 0;
    virtual HRESULT PostWebMessageAsString(LPCWSTR message) = 0;
    virtual HRESULT add_WebMessageReceived(void *unused_31) = 0;
    virtual HRESULT remove_WebMessageReceived(void *unused_32) = 0;
    virtual HRESULT CallDevToolsProtocolMethod(void *unused_33) = 0;
    virtual HRESULT get_BrowserProcessId(void *unused_34) = 0;
    virtual HRESULT get_CanGoBack(void *unused_35) = 0;
    virtual HRESULT get_CanGoForward(void *unused_36) = 0;
    virtual HRESULT GoBack() = 0;
    virtual HRESULT GoForward() = 0;
    virtual HRESULT GetDevToolsProtocolEventReceiver(void *unused_39) = 0;
    virtual HRESULT Stop() = 0;
    virtual HRESULT add_NewWindowRequested(void *unused_41) = 0;
    virtual HRESULT remove_NewWindowRequested(void *unused_42) = 0;
    virtual HRESULT add_DocumentTitleChanged(void *unused_43) = 0;
    virtual HRESULT remove_DocumentTitleChanged(void *unused_44) = 0;
    virtual HRESULT get_DocumentTitle(void *unused_45) = 0;
    virtual HRESULT AddHostObjectToScript(void *unused_46) = 0;
    virtual HRESULT RemoveHostObjectFromScript(void *unused_47) = 0;
    virtual HRESULT OpenDevToolsWindow(void *unused_48) = 0;
    virtual HRESULT add_ContainsFullScreenElementChanged(void *unused_49) = 0;
    virtual HRESULT remove_ContainsFullScreenElementChanged(void *unused_50) = 0;
    virtual HRESULT get_ContainsFullScreenElement(void *unused_51) = 0;
    virtual HRESULT add_WebResourceRequested(void *unused_52) = 0;
    virtual HRESULT remove_WebResourceRequested(void *unused_53) = 0;
    virtual HRESULT AddWebResourceRequestedFilter(void *unused_54) = 0;
    virtual HRESULT RemoveWebResourceRequestedFilter(void *unused_55) = 0;
    virtual HRESULT add_WindowCloseRequested(void *unused_56) = 0;
    virtual HRESULT remove_WindowCloseRequested(void *unused_57) = 0;
};

/* The entry point the loader binds. Everything else is reached through the
   interfaces above. */
HRESULT CreateCoreWebView2EnvironmentWithOptions(
    LPCWSTR browserExecutableFolder, LPCWSTR userDataFolder,
    void *environmentOptions, void *environmentCreatedHandler);

#endif
