#include <stdio.h>
class A { public: int v; A() { v = 1; } };
class B : public A { public: B() { v = 2; } };
class C : public B { public: C() { v = 3; } int get() { return v; } };
int main(void) { C c; printf("%d\n", c.get()); return 0; }
