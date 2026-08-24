#include <stdio.h>
int main(void) {
    const wchar_t *w = L"héllo";
    const char16_t *u = u"ab";
    const char32_t *U = U"ab";
    wchar_t held[8] = L"hi";
    printf("%d %d %d %d %d\n",
           (int)sizeof(L"abc"), (int)sizeof(u"abc"), (int)sizeof(U"abc"),
           (int)w[1], (int)held[1]);
    printf("%d %d %d\n", (int)u[0], (int)U[1], (int)L'x');
    return 0;
}
