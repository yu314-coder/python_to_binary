#include <cstdio>
static int gone = 0;
struct R { int n; R() : n(5) {} ~R() { gone += n; } };
struct A { int v; A() : v(1) {} virtual ~A() { gone += 100; } };
struct B : virtual A { R r; int twice() { return r.n * 2; } };
struct D : B { int d; D() : d(7) {} ~D() { printf("~D %d\n", twice()); } };
int main() {
    { D o; printf("%d %d %d\n", o.v, o.r.n, o.d); }
    printf("gone %d\n", gone);
    { B b; printf("%d %d\n", b.v, b.twice()); }
    printf("gone %d\n", gone);
    return 0;
}
