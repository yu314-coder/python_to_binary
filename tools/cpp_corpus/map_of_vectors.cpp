#include <stdio.h>
#include <map>
#include <string>
#include <vector>
int main(){ std::map<std::string, std::vector<int> > groups;
  groups[std::string("a")].push_back(1);
  groups[std::string("a")].push_back(2);
  groups[std::string("b")].push_back(3);
  printf("%d %d %d\n", (int)groups.size(), (int)groups[std::string("a")].size(),
         groups[std::string("b")][0]); return 0; }
