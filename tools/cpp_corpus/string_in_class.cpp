#include <stdio.h>
#include <string>
class Named { public: std::string name; Named(){ name.assign("none"); }
  void set(const char *s){ name.assign(s); } const char *get(){ return name.c_str(); } };
int main(void){ Named n; n.set("box"); printf("%s\n", n.get()); return 0; }
