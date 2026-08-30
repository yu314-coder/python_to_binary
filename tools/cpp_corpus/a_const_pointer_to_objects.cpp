#include <cstdio>
struct P { int v; P(int x) : v(x) {} int get() const { return v; } };
static int sum(const P *items, int n) { int t = 0; for (int i = 0; i < n; ++i) t += items[i].get(); return t; }
int main() {
    P items[3] = { P(1), P(2), P(3) };
    const P *p = items;
    printf("%d %d %d\n", sum(items, 3), p->get(), p[2].v);
    return 0;
}
