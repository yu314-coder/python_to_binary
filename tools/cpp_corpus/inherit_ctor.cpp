#include <stdio.h>
class Base { public: int v; Base() { v = 1; printf("base\n"); } };
class Derived : public Base { public: int w; Derived() { w = 2; printf("derived\n"); } };
int main(void) { Derived d; printf("%d %d\n", d.v, d.w); return 0; }
