#include <stdio.h>
class Base { public: virtual int f() const { return 1; } virtual ~Base() = default; };
class Sub final : public Base { public: int f() const override { return 2; } };
int main(){ Sub s; Base &r = s; printf("%d\n", r.f()); return 0; }
