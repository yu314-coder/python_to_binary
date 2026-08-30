#include <cstdio>
struct E { int c; E(int v) : c(v) {} };
struct R { int *p; R(int n) { if (n < 0) throw E(n); p = new int(n); } ~R() { delete p; } };
int main() {
    int got = 0, ok = 0;
    { R a(5); ok = *a.p; }
    try { R b(-2); ok += 100; } catch (const E &e) { got = e.c; }
    printf("%d %d\n", ok, got);
    return 0;
}
