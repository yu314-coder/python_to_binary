#include <stdio.h>
class C { public: int a; int b; C(int x, int y) : a(x), b(y) { } };
int main(void){ C c(2, 3); printf("%d\n", c.a + c.b); return 0; }
