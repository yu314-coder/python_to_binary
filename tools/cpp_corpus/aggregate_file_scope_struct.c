#include <stdio.h>
struct P { int x; int y; int z; };
static struct P q = {7,8,9};
int main(){ printf("%d%d%d\n",q.x,q.y,q.z); return 0; }
