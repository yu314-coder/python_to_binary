#include <cstdio>
#include <vector>
int main() {
    std::vector<std::vector<int> > grid;
    for (int r = 0; r < 3; ++r) {
        std::vector<int> row;
        for (int c = 0; c < 3; ++c) row.push_back(r * 3 + c);
        grid.push_back(row);
    }
    int diag = 0;
    for (int i = 0; i < 3; ++i) diag += grid[i][i];
    grid[1][1] = 99;
    printf("%d %d %d %d\n", (int)grid.size(), diag, grid[1][1], grid[2][0]);
    return 0;
}
