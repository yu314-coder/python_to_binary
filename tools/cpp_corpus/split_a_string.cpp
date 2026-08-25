#include <stdio.h>
#include <string>
#include <vector>
int main(){ std::vector<std::string> parts;
  std::string text("a,b,c"); std::string cur;
  for (int i = 0; i < text.size(); i++) {
    if (text[i] == ',') { parts.push_back(cur); cur.clear(); }
    else { cur.push_back(text[i]); } }
  parts.push_back(cur);
  for (size_t i = 0; i < parts.size(); i++) printf("[%s]", parts[i].c_str());
  printf("\n"); return 0; }
