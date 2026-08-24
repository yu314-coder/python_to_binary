#include <stdio.h>
class A { public: int a; A(){a=1;} virtual int who(){return 1;} virtual ~A(){} };
class B : public A { public: int b; B(){b=2;} int who(){return 2;} };
class C : public B { public: int c; C(){c=3;} int who(){return 3;} };
int main(void){ C x; A *p = &x; printf("%d %d %d %d\n", p->who(), x.a, x.b, x.c); return 0; }
