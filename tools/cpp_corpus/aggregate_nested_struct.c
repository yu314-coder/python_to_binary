#include <stdio.h>
struct Inner { int a; int b; };
struct Outer { struct Inner i; int c; };
int main(){ struct Outer o = {{1,2},3}; printf("%d%d%d\n",o.i.a,o.i.b,o.c); return 0; }
