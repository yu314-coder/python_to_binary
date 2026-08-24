#include <stdio.h>
#include <vector>
#include <string>
int main(void){ std::vector<std::string> v;
  std::string a("one"); std::string b("two");
  v.push_back(a); v.push_back(b);
  printf("%s %s %lu\n", v[0].c_str(), v[1].c_str(), (unsigned long)v.size()); return 0; }
