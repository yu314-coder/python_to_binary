#include <stdio.h>
#include <vector>
int main(){ std::vector<std::vector<int> > g; std::vector<int> r; r.push_back(1); r.push_back(2);
  g.push_back(r); printf("%d %d %d\n", (int)g.size(), g[0][0], g[0][1]); return 0; }
