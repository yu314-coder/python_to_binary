#include <cstdio>
struct M { int v; M() { v = 3; } };
struct H { M *p; H(M &m) { p = &m; } int get() { return p->v; } };
int main() { M m; H h(m); printf("%d\n", h.get()); return 0; }
