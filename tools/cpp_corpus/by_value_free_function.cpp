#include <stdio.h>
class V { public: int n; V() { n = 5; } int get() { return n; } };
int twice(V v) { return v.get() * 2; }
int main(void) { V a; printf("%d\n", twice(a)); return 0; }
