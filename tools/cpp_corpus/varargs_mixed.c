#include <stdio.h>
#include <stdarg.h>
static int mixed(int n, ...){ va_list a; int t; va_start(a, n);
  t = va_arg(a, int); t += (int)va_arg(a, double); t += va_arg(a, int);
  va_end(a); return t; }
int main(void){ printf("%d\n", mixed(3, 10, 2.9, 5)); return 0; }
