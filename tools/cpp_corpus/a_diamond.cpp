#include <cstdio>
static int made = 0;
struct A { int v; A(int x) : v(x) { ++made; } virtual int get() { return v; } virtual ~A() {} };
struct B : virtual A { int b; B() : A(1), b(10) {} };
struct C : virtual A { int c; C() : A(2), c(20) {} };
struct D : B, C { int d; D() : A(3), d(30) {} int get() { return v + d; } };
int main() {
    D o;
    A *pa = &o; B *pb = &o; C *pc = &o;
    o.v = 7;
    printf("%d %d %d %d %d %d %d\n", o.v, pb->v, pc->v, pa->get(), o.b, o.c, made);
    B alone;
    printf("%d %d %d\n", alone.v, alone.b, made);
    return 0;
}
