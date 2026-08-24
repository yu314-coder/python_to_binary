#include <stdio.h>
class C { public: int n; C(){n=1;} C(const C &o){ n = o.n + 100; } };
int main(void){ C a; C b(a); printf("%d %d\n", a.n, b.n); return 0; }
