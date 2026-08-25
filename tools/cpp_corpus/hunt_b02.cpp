#include <stdio.h>
class C { public: int n; C() { n = 1; } C &operator=(const C &o) { if (this == &o) return *this; n = o.n; return *this; } };
int main() { C a; C b; a.n = 7; b = a; printf("%d\n", b.n); return 0; }
