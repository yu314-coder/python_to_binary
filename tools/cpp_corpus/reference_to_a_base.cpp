#include <stdio.h>
class A { public: virtual int f() = 0; virtual ~A(){} };
class B : public A { public: int f() override { return 4; } };
int main(){ B b; A &r = b; printf("%d\n", r.f()); return 0; }
