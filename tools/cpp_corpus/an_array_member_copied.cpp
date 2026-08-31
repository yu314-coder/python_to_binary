#include <cstdio>
struct Grid {
    int cells[4];
    int n;
    Grid() : n(0) { for (int i = 0; i < 4; ++i) cells[i] = i; }
    int sum() const { int t = 0; for (int i = 0; i < 4; ++i) t += cells[i]; return t; }
};
static int take(Grid g) { g.cells[0] = 100; return g.sum(); }
int main() {
    Grid a;
    Grid b = a;
    b.cells[1] = 50;
    printf("%d %d %d\n", a.sum(), b.sum(), take(a));
    return 0;
}
