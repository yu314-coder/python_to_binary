#include <stdio.h>
#include <vector>
int main(void){ std::vector<int> a; a.push_back(1);
  std::vector<int> b; b.push_back(2);
  printf("%d %d\n", a[0], b[0]); return 0; }
