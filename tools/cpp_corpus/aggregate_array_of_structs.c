#include <stdio.h>
struct P { int x; int y; };
int main(){ struct P v[3] = {{1,2},{3,4},{5,6}}; printf("%d%d%d\n",v[0].x,v[1].y,v[2].x); return 0; }
