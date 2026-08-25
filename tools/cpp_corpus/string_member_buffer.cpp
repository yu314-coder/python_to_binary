#include <stdio.h>
#include <string>
class Buffer { std::string data;
public: void add(const char *s) { data += s; }
  const char *text() const { return data.c_str(); }
  int size() const { return data.size(); } };
int main(){ Buffer b; b.add("one"); b.add("-two");
  printf("%s %d\n", b.text(), b.size()); return 0; }
