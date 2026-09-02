/* <limits.h>: the widths, which decide what a program may assume. */
#include <limits.h>
#include <stdio.h>

int main(void) {
    printf("%d %d\n", CHAR_BIT, INT_MAX);
    printf("%d\n", INT_MIN + 1);
    printf("%lld\n", (long long)SCHAR_MAX + (long long)UCHAR_MAX);
    printf("%d %d\n", SHRT_MAX, UCHAR_MAX);
    return 0;
}
