#include <stdio.h>
class Inner { public: int v; Inner() { v = 5; } };
class Outer { public: Inner in; };
int main() { Outer o; printf("%d\n", o.in.v); return 0; }
