#include <stdio.h>
struct N { int v; N *next; };
int main(){ N b = {2, 0}; N a = {1, &b}; printf("%d%d\n", a.v, a.next->v); return 0; }
