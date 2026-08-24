#include <stdio.h>
struct P { int x; int y; };
class K { public: int n; K() { n = 5; } int get() { return n; } };
int main(void) { P p; p.x = 1; p.y = 2; K k; printf("%d %d %d\n", p.x, p.y, k.get()); return 0; }
