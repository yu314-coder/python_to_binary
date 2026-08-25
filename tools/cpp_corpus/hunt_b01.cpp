#include <stdio.h>
class Base { public: int n; Base(int v) { n = v; } virtual int show() { return n; } virtual ~Base() {} };
class Sub : public Base { public: Sub(int v) : Base(v * 2) { } int show() { return Base::show() + 1; } };
int main() { Sub s(5); printf("%d\n", s.show()); return 0; }
