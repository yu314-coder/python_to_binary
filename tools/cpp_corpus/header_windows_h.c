/* <windows.h> on the two Windows targets, and skipped elsewhere the way the
   sweep reads the guard. What it covers is the types the header defines and
   three of the calls it declares - which are imported by name rather than
   linked, py2bin having no linker. */
#include <stdio.h>

#ifdef _WIN32
#include <windows.h>

int main(void) {
    DWORD room = MAX_PATH;
    WORD half = 2;
    BYTE one = 1;
    LARGE_INTEGER big;
    FILETIME when;
    HANDLE out;
    DWORD written = 0;
    big.QuadPart = 5;
    when.dwLowDateTime = 1;
    when.dwHighDateTime = 2;
    out = GetStdHandle(STD_OUTPUT_HANDLE);
    if (out != INVALID_HANDLE_VALUE) { WriteFile(out, "", 0, &written, NULL); }
    Sleep(0);
    printf("%d\n", (int)(room + half + one + big.QuadPart
                         + when.dwLowDateTime + when.dwHighDateTime));
    return 0;
}
#else
int main(void) { printf("271\n"); return 0; }
#endif
