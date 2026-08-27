#include <stdio.h>
#include <wchar.h>
int main(void) {
    wchar_t line[64];
    const wchar_t *who = L"world";
    int n = swprintf(line, 64, L"hi %ls #%d", who, 7);
    int i = 0;
    printf("%d:", n);
    while (line[i] != 0) { printf("%c", (char)line[i]); i = i + 1; }
    printf("\n");
    return 0;
}
