#include <stdio.h>
class V { public: int x; V(int a) { x = a; } int get() { return x; } };
int main(void) { V a(1); V b(2); V c(3);
  printf("%d\n", a.get() + b.get() * c.get()); return 0; }
