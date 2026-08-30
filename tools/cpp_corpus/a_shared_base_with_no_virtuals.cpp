#include <cstdio>
static int made = 0;
struct A { int v; A(int x) : v(x) { ++made; } int twice() { return v * 2; } };
struct B : virtual A { int b; B() : A(1), b(5) {} };
struct C : virtual A { int c; C() : A(2), c(6) {} };
struct D : B, C { D() : A(7) {} };
int main() { D o; printf("%d %d %d %d %d\n", o.v, o.twice(), o.b, o.c, made); return 0; }
