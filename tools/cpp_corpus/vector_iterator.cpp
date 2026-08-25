#include <stdio.h>
#include <vector>
int main(){ std::vector<int> v; v.push_back(3); v.push_back(4);
  for (std::vector<int>::iterator it = v.begin(); it != v.end(); ++it) printf("%d",*it);
  printf("\n"); return 0; }
