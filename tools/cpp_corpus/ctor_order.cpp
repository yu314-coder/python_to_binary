#include <stdio.h>
class Inner { public: int v; Inner() { v = 1; printf("inner\n"); } };
class Outer { public: Inner i; int n; Outer() { n = 2; printf("outer\n"); } };
int main(void) { Outer o; printf("%d %d\n", o.i.v, o.n); return 0; }
