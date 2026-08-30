#include <cstdio>
static int built = 0, gone = 0;
struct A { int v; A(int x) : v(x) { ++built; } virtual ~A() { ++gone; } virtual int who() { return 1; } };
struct B : virtual A { B() : A(1) {} };
struct C : virtual A { C() : A(2) {} int who() { return 3; } };
struct D : B, C { D() : A(9) {} };
int main() {
    { D o; printf("%d %d %d\n", o.v, o.who(), built); }
    printf("%d\n", gone);
    { B b; printf("%d %d\n", b.v, b.who()); }
    printf("%d %d\n", built, gone);
    return 0;
}
