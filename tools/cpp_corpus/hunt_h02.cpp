#include <stdio.h>
class H { public: int n; H(int v) { n = v; } H(const H &o) { n = o.n + 100; } };
int use(H h) { return h.n; }
int main() { H a(3); printf("%d %d\n", use(a), a.n); return 0; }
