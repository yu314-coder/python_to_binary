#include <stdio.h>
union U { int i; char c[4]; };
int main(){ union U u = {0x41424344}; printf("%d %c\n", u.i, u.c[0]); return 0; }
