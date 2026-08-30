#include <cstdio>
static int add(int a, int b = 10, int c = 100) { return a + b + c; }
struct S { int n; S(int a = 5) : n(a) {} int get(int m = 2) const { return n * m; } };
int main() {
    S d; S e(7);
    printf("%d %d %d %d %d %d\n", add(1), add(1, 2), add(1, 2, 3), d.n, e.get(), e.get(3));
    return 0;
}
