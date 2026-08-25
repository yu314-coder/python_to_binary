#include <stdio.h>
class C { public: int n; C(int v) { n = v; } C(const C &o) { n = o.n + 100; } };
int main() { C a(3); C b = a; printf("%d %d\n", a.n, b.n); return 0; }
