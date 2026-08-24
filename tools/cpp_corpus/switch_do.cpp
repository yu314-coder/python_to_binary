#include <stdio.h>
int main(void){ int n = 2, t = 0;
  switch (n) { case 1: t = 10; break; case 2: t = 20; break; default: t = 30; }
  int i = 0; do { i++; } while (i < 3);
  printf("%d %d\n", t, i); return 0; }
