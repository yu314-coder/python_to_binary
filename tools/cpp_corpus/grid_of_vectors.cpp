#include <stdio.h>
#include <vector>
class Grid { public: std::vector<std::vector<int> > cells;
  Grid(int n) { for (int i = 0; i < n; i++) { std::vector<int> row;
    for (int j = 0; j < n; j++) row.push_back(i * n + j); cells.push_back(row); } }
  int at(int i, int j) { return cells[i][j]; } };
int main(){ Grid g(3); printf("%d %d\n", g.at(0,1), g.at(2,2)); return 0; }
