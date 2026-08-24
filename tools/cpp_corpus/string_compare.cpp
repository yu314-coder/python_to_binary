#include <stdio.h>
#include <string>
int main(void){ std::string a("abc"); std::string b("abd");
  printf("%d %d %d\n", (int)(a == a), (int)(a != b), a.compare("abc")); return 0; }
