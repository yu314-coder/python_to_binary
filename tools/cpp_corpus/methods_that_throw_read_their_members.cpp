// Methods that throw and read their own members: an array member, a plain one, and one read before the throw.
#include <cstdio>
struct E { int c; E(int v) : c(v) {} };
struct A { int d[4]; int get(int i) { if (i > 3) throw E(i); return d[i]; } };
struct B { int n; int get(int i) { if (i > 3) throw E(i); return n + i; } };
struct C { int n; int get(int i) { if (i > 3) { throw E(i); } return n + i; } };
struct D { int n; int get(int i) { int r = n + i; if (i > 3) throw E(i); return r; } };
int main() {
    A a; a.d[1] = 3; B b; b.n = 4; C c; c.n = 5; D d; d.n = 6;
    int t = 0;
    try { t = a.get(1) + b.get(1) + c.get(1) + d.get(1); } catch (const E &e) { t = -1; }
    printf("%d\n", t);
    return 0;
}
