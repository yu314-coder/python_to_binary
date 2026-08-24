#include <stdio.h>
int main(void){ int a = 3, b = 7;
  int m = a < b ? a : b;
  int t = (a > 0 && b > 0) || (a == b);
  printf("%d %d %d\n", m, t, !!(a & 1)); return 0; }
