#include <stdio.h>
struct Inner { int a; int b; };
class Outer { public: struct Inner held; Outer(){ held.a = 1; held.b = 2; }
  int sum() const { return held.a + held.b; } };
int main(){ Outer o; printf("%d\n", o.sum()); return 0; }
