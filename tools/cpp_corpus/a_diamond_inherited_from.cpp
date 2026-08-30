#include <cstdio>
static int made = 0;
struct A { int v; A(int x) : v(x) { ++made; } virtual int get() { return v; } };
struct B : virtual A { B() : A(1) {} };
struct C : virtual A { C() : A(2) {} };
struct D : B, C { D() : A(3) {} int get() { return v * 2; } };
struct E : D { E() : A(4) {} int get() { return v * 100; } };
int main() {
    E e;
    A *pa = &e; D *pd = &e; B *pb = &e;
    printf("%d %d %d %d %d\n", e.v, pa->get(), pd->get(), pb->v, made);
    return 0;
}
