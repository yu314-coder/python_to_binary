#include <cstdio>
struct E { int c; E(int v) : c(v) {} };
static int inner(int n) { if (n < 0) throw E(n); return n; }
static int middle(int n) { try { return inner(n); } catch (const E &e) { throw; } }
int main() {
    int got = 0;
    try { middle(-7); } catch (const E &e) { got = e.c; }
    printf("%d %d\n", middle(4), got);
    return 0;
}
