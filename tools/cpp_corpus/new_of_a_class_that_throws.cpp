// `new R(-5)` whose constructor throws assigns nothing; the handler runs and the pointer stays null.
#include <cstdio>
struct E { int c; E(int v) : c(v) {} };
static int built = 0;
struct R { int v; R(int x) : v(x) { if (x < 0) throw E(x); built++; } };
int main() {
    R *ok = nullptr, *bad = nullptr;
    try { ok = new R(1); bad = new R(-5); } catch (const E &e) { printf("caught %d\n", e.c); }
    printf("%d %d %d\n", ok != nullptr, bad == nullptr, built);
    delete ok;
    return 0;
}
