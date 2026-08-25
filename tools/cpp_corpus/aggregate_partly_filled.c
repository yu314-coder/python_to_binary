#include <stdio.h>
struct P { int x; int y; int z; };
int main(){ struct P p = {1}; printf("%d%d%d\n",p.x,p.y,p.z); return 0; }
