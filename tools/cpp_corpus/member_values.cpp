#include <stdio.h>
class C {
public:
  int a = 1;
  int b = 2;
  const char *name = "unset";
  C() {}
  C(int v) : b(v) {}
  int sum() const { return a + b; }
};
int main(){ C x; C y(9); printf("%d %d %s\n", x.sum(), y.sum(), x.name); return 0; }
