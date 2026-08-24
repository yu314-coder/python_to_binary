#include <stdio.h>
#include <string>
int main(void){ std::string a("ab"); std::string b("cd");
  a.append(b); printf("%s %d %d\n", a.c_str(), (int)a.size(), (int)(a == a)); return 0; }
