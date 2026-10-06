// A lambda whose body holds a try and a throw of its own.
#include <cstdio>
struct E { int c; E(int v) : c(v) {} };
int main() {
    auto safe = [](int x) { try { if (x < 0) throw E(x); return x; } catch (const E &e) { return -e.c * 100; } };
    printf("%d %d\n", safe(4), safe(-2));
    return 0;
}
