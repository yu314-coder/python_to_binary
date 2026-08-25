#include <stdio.h>
#include <set>
int main(){ std::set<int> s; s.insert(3); s.insert(1); s.insert(3);
  printf("%d %d %d\n", (int)s.size(), (int)s.count(1), (int)s.count(9)); return 0; }
