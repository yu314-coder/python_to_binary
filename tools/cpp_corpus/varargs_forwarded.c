#include <stdio.h>
#include <stdarg.h>
static int inner(int n, va_list a){ int t = 0; int i; for (i = 0; i < n; i++) t += va_arg(a, int); return t; }
static int outer(int n, ...){ va_list a; int t; va_start(a, n); t = inner(n, a); va_end(a); return t; }
int main(void){ printf("%d\n", outer(3, 4, 5, 6)); return 0; }
