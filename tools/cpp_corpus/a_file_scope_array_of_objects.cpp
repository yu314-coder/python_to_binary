#include <cstdio>
static int made = 0;
struct Cell { int v; Cell() : v(0) { ++made; } Cell(int x) : v(x) { ++made; } };
static Cell table[3];
int main() {
    Cell local[2];
    local[0].v = 5;
    Cell listed[3] = { Cell(1), Cell(2), Cell(3) };
    table[1].v = 7;
    printf("%d %d %d %d %d\n", local[0].v, listed[2].v, table[1].v, table[0].v, made);
    return 0;
}
