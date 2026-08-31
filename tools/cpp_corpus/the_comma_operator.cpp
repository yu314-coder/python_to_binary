#include <cstdio>
struct V { int x; V(int v) : x(v) {} };
static int pick(int a) { return a > 2 ? (a > 4 ? 100 : 50) : 10; }
int main() {
    int i = 0, j = 0;
    for (i = 0, j = 10; i < 3; ++i, --j) { }
    int k = (i + 1, j + 2);
    V a(1), b(2);
    const V &c = a.x < b.x ? b : a;
    printf("%d %d %d %d %d\n", i, j, k, pick(5), c.x);
    return 0;
}
