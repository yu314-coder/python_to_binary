#include <cstdio>
struct E { int c; E(int v) : c(v) {} };
struct A { virtual int go(int n) { if (n < 0) throw E(n); return n * 2; } };
struct B : A { int go(int n) { return A::go(n) + 1; } };
int main() {
    B b; A *p = &b; int got = 0; int ok = p->go(3);
    try { p->go(-1); } catch (const E &e) { got = e.c; }
    printf("%d %d\n", ok, got);
    return 0;
}
