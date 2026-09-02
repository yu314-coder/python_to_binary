#include <cstdio>
struct D4 { struct Mid { struct Bot { int v; } b; } m; };
int main() { D4 d; d.m.b.v = 7; printf("%d\n", d.m.b.v); return 0; }
