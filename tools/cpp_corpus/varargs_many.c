#include <stdio.h>
#include <stdarg.h>
static int many(int n, ...){ va_list a; int t = 0; int i; va_start(a, n);
  for (i = 0; i < n; i++) t += va_arg(a, int); va_end(a); return t; }
int main(void){ printf("%d\n", many(12, 1,2,3,4,5,6,7,8,9,10,11,12)); return 0; }
