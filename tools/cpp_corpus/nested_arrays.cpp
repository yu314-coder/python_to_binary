#include <stdio.h>
class E { public: int n; E() { n = 2; } int get() { return n; } };
int main(void) { E e[3]; int total = 0; int i;
  for (i = 0; i < 3; i++) { if (e[i].get() > 1) { total = total + e[i].get(); } }
  printf("%d\n", total); return 0; }
