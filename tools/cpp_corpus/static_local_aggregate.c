#include <stdio.h>
struct P { int x; int y; };
static int used(void){ static struct P p = {3, 4}; static int t[3] = {1,2,3};
  p.x++; return p.x + p.y + t[2]; }
int main(void){ used(); printf("%d\n", used()); return 0; }
