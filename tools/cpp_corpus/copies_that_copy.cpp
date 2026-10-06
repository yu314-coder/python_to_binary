// A copy is made by the class's own code wherever C++ makes one. A struct
// holding a class, a class deriving from one, a value handed back from a
// parameter, a global and a member: each was copied as bytes, or built and
// then assigned, and a class that counts its copies said so differently.
#include <cstdio>
#include <vector>
struct C {
    int v;
    C() : v(0) {}
    C(int v) : v(v) {}
    C(const C &o) : v(o.v) { printf("copy %d\n", o.v); }
    C &operator=(const C &o) { v = o.v; printf("assign %d\n", o.v); return *this; }
};
struct Agg { C c; int n; };
struct Nested { Agg a; C extra; };
struct D : C { int extra; };
struct W { C c; W(const C &x) : c(x) {} };
struct H { C c; C get() const { return c; } };
C global(7);
C pass(C p) { return p; }
C get_global() { return global; }
struct Bag { std::vector<int> items; int n; };
int main() {
    C a(1);
    printf("-- aggregate copy\n"); Agg g; g.c = a; g.n = 2; Agg g2 = g;
    printf("-- aggregate assign\n"); Agg g3; g3.c = C(9); g3 = g;
    printf("-- nested\n"); Nested n1; n1.a.c = C(3); n1.extra = C(4); Nested n2 = n1;
    printf("-- derived\n"); D d; d.v = 5; d.extra = 6; D e = d;
    printf("-- member from a reference\n"); W w(a);
    printf("-- returns\n"); C r1 = pass(a); C r2 = get_global(); H h; h.c = C(8); C r3 = h.get();
    printf("-- a struct holding a vector\n");
    Bag b; b.items.push_back(1); b.n = 1;
    Bag c = b; c.items[0] = 5; c.items.push_back(6);
    Bag e2; e2 = b; e2.items[0] = 9;
    printf("%d %d %d %d %d %d %d %d %d %d %d\n", g2.c.v, g3.c.v, n2.a.c.v, n2.extra.v,
           e.v, e.extra, w.c.v, r1.v, r2.v, r3.v, (int)b.items.size());
    printf("%d %d %d %d\n", b.items[0], c.items[0], (int)c.items.size(), e2.items[0]);
    return 0;
}
