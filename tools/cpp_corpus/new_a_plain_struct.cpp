#include <cstdio>
struct P { int x; int y; };
int main() { P *p = new P(); p->x = 3; printf("%d\n", p->x); delete p; return 0; }
