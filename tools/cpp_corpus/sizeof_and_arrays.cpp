#include <stdio.h>
struct P { int x; int y; };
int main(void){ P grid[3]; grid[0].x = 1;
  printf("%d %d %d\n", (int)sizeof(P), (int)(sizeof(grid)/sizeof(grid[0])), grid[0].x); return 0; }
