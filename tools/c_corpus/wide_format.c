#include <stdio.h>
#include <wchar.h>
int main(void) {
    wchar_t line[64];
    int n = swprintf(line, 64, L"n=%d s=%s x=%04X", 42, "abc", 255);
    int i = 0;
    printf("%d:", n);
    while (line[i] != 0) { printf("%c", (char)line[i]); i = i + 1; }
    printf("\n");
    return 0;
}
