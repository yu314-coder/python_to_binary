#include <stdio.h>
class C { public: int v; C(int x); int get(); };
C::C(int x) { v = x; }
int C::get() { return v; }
int main(){ C c(7); printf("%d\n", c.get()); return 0; }
