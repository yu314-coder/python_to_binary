#include <stdio.h>
struct A { int a; A(){a=1;} int get(){return a;} };
struct B : A { int b; B(){b=2;} };
int main(void){ B x; printf("%d %d\n", x.get(), x.b); return 0; }
