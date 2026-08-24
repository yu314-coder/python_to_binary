#include <stdio.h>
class M { public: int base; M() { base = 10; }
  int two(int a, int b) { return base + a + b; }
  int three(int a, int b, int c) { return two(a, b) + c; } };
int main(void) { M m; printf("%d %d\n", m.two(1, 2), m.three(1, 2, 3)); return 0; }
