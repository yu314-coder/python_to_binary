#include <stdio.h>
struct P { int x, y; };
int main(){ P a{1,2}; P b = {3,4}; printf("%d%d%d%d\n",a.x,a.y,b.x,b.y); return 0; }
