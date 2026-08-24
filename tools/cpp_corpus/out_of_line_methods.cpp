#include <stdio.h>
class C { public: int n; C(); int twice(); int plus(int k); };
C::C() { n = 6; }
int C::twice() { return n * 2; }
int C::plus(int k) { return n + k; }
int main(void){ C c; printf("%d %d\n", c.twice(), c.plus(4)); return 0; }
