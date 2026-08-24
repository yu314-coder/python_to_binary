#include <stdio.h>
int main(void){ int t = 0;
  for (int i = 0; i < 10; i++) { if (i % 2) continue; if (i > 6) break; t += i; }
  printf("%d\n", t); return 0; }
