#include <cstdio>
#include <vector>
struct Cell { int v; Cell() : v(0) {} Cell(int x) : v(x) {} };
int main() {
    std::vector<Cell> v;
    for (int i = 0; i < 40; ++i) v.push_back(Cell(i));
    int total = 0;
    for (size_t i = 0; i < v.size(); ++i) total += v[i].v;
    printf("%d %d %d %d\n", (int)v.size(), v[0].v, v[39].v, total);
    return 0;
}
