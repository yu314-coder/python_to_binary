#include <stdio.h>
#include <vector>

struct Cell { int n; Cell(int v) : n(v) {} int get() const { return n; } };

int main() {
    std::vector<Cell *> cells;
    cells.push_back(new Cell(7));
    cells.push_back(new Cell(11));
    int total = 0;
    for (unsigned i = 0; i < cells.size(); i++) { total += cells[i]->get(); }
    printf("%d\n", total);
    return 0;
}
