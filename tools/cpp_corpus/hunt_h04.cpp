#include <stdio.h>
struct P { int x, y; void set() { x = 1; y = 2; } };
int main() { P p; p.set(); printf("%d %d\n", p.x, p.y); return 0; }
