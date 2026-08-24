#include <stdio.h>
#include <vector>
int main(void){ std::vector<int> v; v.push_back(1); v.push_back(2);
  int t = 0; for (int x : v) { t += x; } printf("%d\n", t); return 0; }
