#include <cstdio>
#include <typeinfo>
struct A { virtual ~A() {} };
struct B : A {};
int main() { B b; A *p = &b; printf("%d\n", typeid(*p) == typeid(B)); return 0; }
