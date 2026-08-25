#include <stdio.h>
struct P { int x; int y; };
static struct P v[2] = {{1,2},{3,4}};
static struct P one = {5};
int main(){ printf("%d%d%d%d%d\n",v[0].x,v[0].y,v[1].x,v[1].y,one.y); return 0; }
