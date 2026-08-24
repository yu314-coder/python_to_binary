#include <stdio.h>
class C { public: int n; C() { n = 100; } int mix(int n) { return this->n + n; } };
int main(void) { C c; printf("%d\n", c.mix(5)); return 0; }
