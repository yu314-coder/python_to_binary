#include <stdio.h>
#include <functional>
#include <vector>
int main(){ std::vector<std::function<int(int)> > all;
  std::function<int(int)> a = [](int x){ return x + 1; };
  std::function<int(int)> b = [](int x){ return x * 10; };
  all.push_back(a); all.push_back(b);
  printf("%d %d\n", all[0](5), all[1](5)); return 0; }
