#include <stdio.h>
int main(void) {
    const wchar_t *w = L"A\U0001F600B";
    int n = 0;
    while (w[n] != 0) n++;
    printf("units=%d size=%d first=%d\n", n, (int)sizeof(L"A\U0001F600B"), (int)w[1]);
    const char *eight = u8"héllo";
    printf("%s %d\n", eight, (int)sizeof(u8"é"));
    return 0;
}
