/* <stdarg.h>: a va_list walked twice, with different counts. */
#include <stdarg.h>
#include <stdio.h>

static int total(int count, ...) {
    va_list rest;
    int sum = 0;
    int i;
    va_start(rest, count);
    for (i = 0; i < count; i++) { sum += va_arg(rest, int); }
    va_end(rest);
    return sum;
}

int main(void) {
    printf("%d\n", total(3, 1, 2, 3));
    printf("%d\n", total(5, 10, 20, 30, 40, 50));
    return 0;
}
