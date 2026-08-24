#include <stdio.h>
class N { public: int v; N(int x) { v = x; }
  int add(int k) { return v + k; } int mul(int k) { return v * k; }
  int both(int k) { return add(k) + mul(k); } };
int main(void) { N n(6); printf("%d %d %d\n", n.add(4), n.mul(3), n.both(2)); return 0; }
