#include <stdio.h>
class A { public: virtual int id() { return 1; } virtual ~A() {} };
class B : public A { public: };
class C : public B { public: int id() { return 3; } };
int main() { C c; B *p = &c; printf("%d\n", p->id()); return 0; }
