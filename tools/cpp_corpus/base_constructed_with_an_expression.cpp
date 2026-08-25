#include <stdio.h>
class A { public: int n; A(int v) : n(v) {} };
class B : public A { public: B(int v) : A(v * 2) {} };
int main(){ B b(3); printf("%d\n", b.n); return 0; }
