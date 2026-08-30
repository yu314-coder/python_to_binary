#include <cstdio>
static int live = 0;
struct G { G() { ++live; } ~G() { --live; } };
struct E { int c; E(int v) : c(v) {} };
static void deep(int n) { G g; if (n < 0) throw E(n); }
int main() {
    int got = 0;
    { G a; deep(1); }
    try { G b; deep(-1); } catch (const E &e) { got = e.c; }
    printf("%d %d\n", got, live);
    return 0;
}
