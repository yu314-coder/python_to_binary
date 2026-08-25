#include <stdio.h>
#include <string>
#include <vector>
int main(){ std::vector<std::string> v; v.push_back(std::string("a"));
  for (size_t i = 0; i < v.size(); i++) printf("%s", v[i].c_str()); printf("\n"); return 0; }
