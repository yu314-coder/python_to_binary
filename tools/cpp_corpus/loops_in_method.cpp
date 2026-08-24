#include <stdio.h>
class S { public: int total; S() { total = 0; }
  void add(int n) { int i; for (i = 1; i <= n; i++) { total = total + i; } }
  int get() { return total; } };
int main(void) { S s; s.add(10); printf("%d\n", s.get()); return 0; }
