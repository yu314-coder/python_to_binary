/* The desktop, clipboard, GDI, global-memory and DPAPI imports, each
   declared the way Windows declares it and called the way a program calls
   it. Written out here rather than included, so the program needs no SDK:
   what is being checked is that py2bin turns each prototype into an import
   of the right library, which is the same question either way. */
#include <stdio.h>

#ifdef _WIN32
typedef void *HANDLE;
typedef void *HWND;
typedef void *HDC;
typedef void *HBITMAP;
typedef void *HGLOBAL;
typedef void *HGDIOBJ;
typedef void *LPVOID;
typedef unsigned int UINT;
typedef unsigned long DWORD;
typedef int BOOL;
typedef unsigned long long SIZE_T;

struct KEYINPUT { unsigned short wVk; unsigned short wScan; DWORD dwFlags; };
struct INPUTBLOCK { DWORD type; struct KEYINPUT ki; DWORD pad[2]; };
struct CRYPTBLOB { DWORD cbData; unsigned char *pbData; };

UINT SendInput(UINT cInputs, struct INPUTBLOCK *pInputs, int cbSize);
BOOL SetCursorPos(int x, int y);
BOOL OpenClipboard(HWND owner);
BOOL CloseClipboard(void);
BOOL EmptyClipboard(void);
HANDLE GetClipboardData(UINT format);
HANDLE SetClipboardData(UINT format, HANDLE held);
HDC GetDC(HWND window);
int ReleaseDC(HWND window, HDC dc);

BOOL BitBlt(HDC dc, int x, int y, int cx, int cy, HDC from, int x1, int y1,
            DWORD rop);
HBITMAP CreateCompatibleBitmap(HDC dc, int cx, int cy);
HDC CreateCompatibleDC(HDC dc);
BOOL DeleteDC(HDC dc);
BOOL DeleteObject(HGDIOBJ held);
HGDIOBJ SelectObject(HDC dc, HGDIOBJ held);

HGLOBAL GlobalAlloc(UINT flags, SIZE_T bytes);
HGLOBAL GlobalFree(HGLOBAL block);
LPVOID GlobalLock(HGLOBAL block);
BOOL GlobalUnlock(HGLOBAL block);
HANDLE LocalFree(HANDLE block);
BOOL GetComputerNameW(void *buffer, DWORD *size);
DWORD GetEnvironmentVariableW(const void *name, void *buffer, DWORD size);

BOOL CryptProtectData(struct CRYPTBLOB *in, const void *why,
                      struct CRYPTBLOB *entropy, void *reserved,
                      void *prompt, DWORD flags, struct CRYPTBLOB *out);
BOOL CryptUnprotectData(struct CRYPTBLOB *in, void **why,
                        struct CRYPTBLOB *entropy, void *reserved,
                        void *prompt, DWORD flags, struct CRYPTBLOB *out);

void *SHCreateMemStream(const unsigned char *bytes, UINT count);

static int act(void) {
    struct INPUTBLOCK input;
    HDC screen;
    HDC copy;
    HBITMAP bitmap;
    HGLOBAL block;
    LPVOID held;
    struct CRYPTBLOB in;
    struct CRYPTBLOB out;
    unsigned char bytes[4];
    DWORD room = 0;
    int total = 0;

    input.type = 1;
    input.ki.wVk = 16;
    input.ki.dwFlags = 2;
    total += (int)SendInput(1, &input, (int)sizeof(input));
    total += SetCursorPos(10, 10);

    if (OpenClipboard(0)) {
        EmptyClipboard();
        block = GlobalAlloc(2, 8);
        held = GlobalLock(block);
        if (held != 0) { GlobalUnlock(block); }
        SetClipboardData(1, block);
        GetClipboardData(1);
        CloseClipboard();
        GlobalFree(block);
    }

    screen = GetDC(0);
    copy = CreateCompatibleDC(screen);
    bitmap = CreateCompatibleBitmap(screen, 4, 4);
    SelectObject(copy, bitmap);
    BitBlt(copy, 0, 0, 4, 4, screen, 0, 0, 13369376);
    DeleteObject(bitmap);
    DeleteDC(copy);
    ReleaseDC(0, screen);

    bytes[0] = 1; bytes[1] = 2; bytes[2] = 3; bytes[3] = 4;
    in.cbData = 4;
    in.pbData = bytes;
    out.cbData = 0;
    out.pbData = 0;
    if (CryptProtectData(&in, 0, 0, 0, 0, 0, &out)) {
        struct CRYPTBLOB back;
        back.cbData = 0;
        back.pbData = 0;
        if (CryptUnprotectData(&out, 0, 0, 0, 0, 0, &back)) {
            total += (int)back.cbData;
            LocalFree(back.pbData);
        }
        LocalFree(out.pbData);
    }

    SHCreateMemStream(bytes, 4);
    GetComputerNameW(0, &room);
    GetEnvironmentVariableW(0, 0, 0);
    return total;
}
#else
static int act(void) { return 0; }
#endif

int main(void) {
    printf("%d\n", act() >= 0);
    return 0;
}
