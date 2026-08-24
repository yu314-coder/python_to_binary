#include <stdio.h>
class A { public: int who() { return 1; } };
class B : public A { public: int who() { return 2; } };
int main(void) { B b; A a; printf("%d %d\n", b.who(), a.who()); return 0; }
