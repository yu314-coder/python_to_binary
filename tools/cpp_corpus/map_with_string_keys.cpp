#include <stdio.h>
#include <map>
#include <string>
int main(){ std::map<std::string,int> m; m[std::string("a")] = 1; m[std::string("b")] = 2;
  printf("%d %d %d %d\n", (int)m.size(), m[std::string("a")], (int)m.count(std::string("b")), (int)m.count(std::string("z"))); return 0; }
