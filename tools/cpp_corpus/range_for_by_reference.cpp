#include <stdio.h>
#include <vector>
int main(){ std::vector<int> v; v.push_back(1); v.push_back(2);
  for (auto &x : v) { x = x * 10; }
  printf("%d %d\n", v[0], v[1]); return 0; }
