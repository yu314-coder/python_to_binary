// An argument that throws is worked out before the call it is handed to: g, which can throw too, is not called with f(-4) in flight.
#include <cstdio>
static int calls = 0;
struct E { int c; E(int v) : c(v) {} };
int f(int x) { calls++; if (x < 0) throw E(x); return x * 10; }
int g(int a, int b) { calls++; if (a + b < 0) throw E(a + b); return a + b; }
int main() {
    int got = 0, r = 0;
    try { r = g(f(1), f(2)); r += g(f(3), f(-4)); printf("not reached\n"); } catch (const E &e) { got = e.c; }
    printf("%d %d %d\n", r, got, calls);
    return 0;
}
