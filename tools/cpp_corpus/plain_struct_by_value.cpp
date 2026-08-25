#include <stdio.h>
struct Point { int x; int y; };
static Point add(Point a, Point b){ Point r; r.x = a.x + b.x; r.y = a.y + b.y; return r; }
int main(){ Point a = {1,2}; Point b = {3,4}; Point c = add(a, b);
  printf("%d %d\n", c.x, c.y); return 0; }
