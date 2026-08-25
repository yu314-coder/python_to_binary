#include <stdio.h>
#include <stdarg.h>
static double average(int n, ...){ va_list a; double t = 0.0; int i;
  va_start(a, n); for (i = 0; i < n; i++) { t += va_arg(a, double); } va_end(a);
  return t / (double)n; }
int main(void){ printf("%.3f\n", average(3, 1.0, 2.0, 6.0)); return 0; }
