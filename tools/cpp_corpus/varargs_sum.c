#include <stdio.h>
#include <stdarg.h>
static int total(int n, ...){ va_list a; int t = 0; int i;
  va_start(a, n); for (i = 0; i < n; i++) { t += va_arg(a, int); } va_end(a); return t; }
int main(void){ printf("%d\n", total(3, 1, 2, 4)); return 0; }
