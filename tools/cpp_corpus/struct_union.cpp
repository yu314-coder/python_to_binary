#include <stdio.h>
struct P { int x; int y; };
union U { int i; float f; };
int main(void){ P p; p.x = 1; p.y = 2; U u; u.i = 65; printf("%d %d\n", p.x + p.y, u.i); return 0; }
