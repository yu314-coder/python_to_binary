#include <stdio.h>
class B;
class A { public: int n; A(){n=1;} int use(); };
class B { public: int m; B(){m=41;} };
int A::use() { B b; return n + b.m; }
int main(void){ A a; printf("%d\n", a.use()); return 0; }
