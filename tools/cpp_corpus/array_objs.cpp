#include <stdio.h>
class Cell { public: int n; Cell() { n = 3; } int get() { return n; } };
int main(void) { Cell c[4]; int t = 0; int i;
  for (i = 0; i < 4; i++) { t = t + c[i].get(); }
  printf("%d\n", t); return 0; }
