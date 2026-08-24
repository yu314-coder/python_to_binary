#include <stdio.h>
class A { public: int v; A() { v = 3; } int get() { return v; } };
class B { public: A inner; int n; B() { n = 1; } int deep() { return inner.get() + n; } };
int main(void) { B b; printf("%d %d\n", b.deep(), b.inner.get()); return 0; }
