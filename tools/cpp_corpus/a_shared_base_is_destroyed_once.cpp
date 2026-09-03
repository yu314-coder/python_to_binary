#include <cstdio>
static int built = 0, gone = 0;
struct A { int v; A() : v(1) { ++built; } virtual ~A() { ++gone; } };
struct B : virtual A { int b; B() : b(2) {} ~B() { printf("~B\n"); } };
struct C : virtual A { int c; C() : c(3) {} ~C() { printf("~C\n"); } };
struct D : B, C { int d; D() : d(4) {} ~D() { printf("~D\n"); } };
int main() {
    { D o; printf("%d %d %d %d\n", o.v, o.b, o.c, o.d); }
    printf("built %d gone %d\n", built, gone);
    { B alone; printf("%d %d\n", alone.v, alone.b); }
    printf("built %d gone %d\n", built, gone);
    A *p = new D();
    delete p;
    printf("built %d gone %d\n", built, gone);
    return 0;
}
