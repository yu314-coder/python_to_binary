#include <cstdio>
struct Bad { int code; Bad(int c) : code(c) {} };
static int risky(int n) { if (n < 0) throw Bad(n); return n * 2; }
int main() {
    int ok = risky(4);
    int caught = 0;
    try { risky(-3); } catch (const Bad &b) { caught = b.code; }
    printf("%d %d\n", ok, caught);
    return 0;
}
