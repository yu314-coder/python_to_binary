#include <stdio.h>
class T { public: int n; int a; int v; T() { n = 1; a = 2; v = 3; }
  void show() { printf("n=%d a=%d v=%d\ttab\n", n, a, v); } };
int main(void) { T t; t.show(); printf("has n and a and v inside\n"); return 0; }
