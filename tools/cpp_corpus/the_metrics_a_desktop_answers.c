/* The constants a companion that shares a desktop names: the system metrics,
   the raster operations BitBlt is told to do, the clipboard's formats, the
   flags GlobalAlloc and CryptProtectData take, what IStream::Stat is asked
   for, and the keys SendInput is given.

   py2bin ships its own <windows.h>, because the SDK's headers demand GCC or
   MSVC, and that header carried the *calls* and almost none of the names they
   are called with - so a program that asks where the desktop begins, or tells
   BitBlt to copy, named a constant nothing declared and the build stopped on
   a line that is correct Windows C.

   Nothing here calls into the desktop; the sweep does not run a Windows
   build. What it asks is that each name is declared and carries the number
   Windows gives it, which is checked against the same sum written out. */
#include <stdio.h>

#ifdef _WIN32
#include <windows.h>

int main(void) {
    long metrics = SM_XVIRTUALSCREEN + SM_YVIRTUALSCREEN
                 + SM_CXVIRTUALSCREEN + SM_CYVIRTUALSCREEN
                 + SM_CMONITORS + SM_CXSMICON + SM_CYSMICON
                 + SM_CXFRAME + SM_CYFRAME + SM_CYCAPTION;
    long raster = (long)SRCCOPY + PATCOPY + DSTINVERT + WHITENESS
                + (long)CAPTUREBLT / 16 + DIB_RGB_COLORS + BI_RGB;
    long clipboard = CF_TEXT + CF_BITMAP + CF_DIB + CF_UNICODETEXT + CF_HDROP;
    long memory = GMEM_FIXED + GMEM_MOVEABLE + GMEM_ZEROINIT + GHND + GPTR;
    long guarded = CRYPTPROTECT_UI_FORBIDDEN + CRYPTPROTECT_LOCAL_MACHINE
                 + STATFLAG_DEFAULT + STATFLAG_NONAME + STATFLAG_NOOPEN;
    long keys = VK_BACK + VK_TAB + VK_RETURN + VK_SHIFT + VK_CONTROL + VK_MENU
              + VK_ESCAPE + VK_SPACE + VK_PRIOR + VK_NEXT + VK_END + VK_HOME
              + VK_LEFT + VK_UP + VK_RIGHT + VK_DOWN + VK_INSERT + VK_DELETE
              + VK_LWIN + VK_F1 + VK_F12 + VK_CAPITAL
              + VK_OEM_1 + VK_OEM_PLUS + VK_OEM_COMMA + VK_OEM_MINUS
              + VK_OEM_PERIOD + VK_OEM_2 + VK_OEM_3 + VK_OEM_4 + VK_OEM_5
              + VK_OEM_6 + VK_OEM_7;
    /* Each sum against the number Windows gives it, checked where it can be:
       a negative array size is not a program, so a wrong value here stops the
       build rather than printing something nobody can compare. The sweep does
       not run a Windows binary, so this is the only place the *values* are
       asked about rather than the names. */
    char settled[(SM_XVIRTUALSCREEN == 76 && SM_CYVIRTUALSCREEN == 79
                  && SRCCOPY == 0x00CC0020L && CAPTUREBLT == 0x40000000L
                  && CF_UNICODETEXT == 13 && GMEM_MOVEABLE == 0x0002
                  && GHND == 0x0042 && CRYPTPROTECT_UI_FORBIDDEN == 0x1
                  && STATFLAG_NONAME == 1 && VK_RETURN == 0x0D
                  && VK_OEM_7 == 0xDE && BI_RGB == 0) ? 1 : -1];
    int width = GetSystemMetrics(SM_CXVIRTUALSCREEN);
    int height = GetSystemMetrics(SM_CYVIRTUALSCREEN);
    settled[0] = 0;
    printf("%ld %ld %ld %ld %ld %ld %d\n", metrics, raster, clipboard,
           memory, guarded, keys, (width >= 0 && height >= 0) + settled[0]);
    return 0;
}
#else
int main(void) {
    long metrics = 76 + 77 + 78 + 79 + 80 + 49 + 50 + 32 + 33 + 4;
    long raster = 0x00CC0020L + 0x00F00021L + 0x00550009L + 0x00FF0062L
                + 0x40000000L / 16 + 0 + 0;
    long clipboard = 1 + 2 + 8 + 13 + 15;
    long memory = 0x0000 + 0x0002 + 0x0040 + 0x0042 + 0x0040;
    long guarded = 0x1 + 0x4 + 0 + 1 + 2;
    long keys = 0x08 + 0x09 + 0x0D + 0x10 + 0x11 + 0x12
              + 0x1B + 0x20 + 0x21 + 0x22 + 0x23 + 0x24
              + 0x25 + 0x26 + 0x27 + 0x28 + 0x2D + 0x2E
              + 0x5B + 0x70 + 0x7B + 0x14
              + 0xBA + 0xBB + 0xBC + 0xBD
              + 0xBE + 0xBF + 0xC0 + 0xDB + 0xDC
              + 0xDD + 0xDE;
    printf("%ld %ld %ld %ld %ld %ld %d\n", metrics, raster, clipboard,
           memory, guarded, keys, 1);
    return 0;
}
#endif
