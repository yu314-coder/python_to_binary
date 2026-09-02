/* <wchar.h>: the three wide-string routines the header carries. `wchar_t` is
   two bytes on Windows and four everywhere else, which is why only its
   contents are printed here and not its width. */
#include <wchar.h>
#include <stdio.h>

int main(void) {
    wchar_t held[8];
    wchar_t copy[8];
    held[0] = L'a'; held[1] = L'b'; held[2] = L'c'; held[3] = 0;
    wcscpy(copy, held);
    printf("%d\n", (int)wcslen(held));
    printf("%d %d\n", (int)held[1], wcscmp(held, copy));
    printf("%d\n", (int)WEOF);
    return 0;
}
