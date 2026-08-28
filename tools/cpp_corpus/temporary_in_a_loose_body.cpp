#include <cstdio>
struct Cell { int v; Cell() : v(0) {} Cell(int n) : v(n) {} int get() const { return v; } };
int main() {
    Cell grid[3];
    for (int i = 0; i < 3; i++) grid[i] = Cell(i * 10);
    int flat[2][3] = { {1,2,3}, {4,5,6} };
    int *p = &flat[1][0];
    printf("%d %d %d %d %d\n", grid[0].get(), grid[2].get(), flat[0][2], p[2], *(p + 1));
    return 0;
}
