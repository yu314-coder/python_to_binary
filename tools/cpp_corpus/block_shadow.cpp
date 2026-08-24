#include <stdio.h>
class K { public: int n; K(int a) { n = a; } int get() { return n; } };
int main(void) { K k(1);
  { K k2(2); printf("%d %d\n", k.get(), k2.get()); }
  printf("%d\n", k.get()); return 0; }
