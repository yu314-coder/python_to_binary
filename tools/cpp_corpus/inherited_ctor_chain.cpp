#include <stdio.h>
int made = 0;
class A { public: A(){ made = made * 10 + 1; } virtual ~A(){} };
class B : public A { public: B(){ made = made * 10 + 2; } };
class C : public B { public: C(){ made = made * 10 + 3; } };
int main(void){ C c; printf("%d\n", made); return 0; }
