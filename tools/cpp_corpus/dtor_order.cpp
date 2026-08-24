#include <stdio.h>
class A { public: int id; A() { id = 0; } ~A() { printf("~A\n"); } };
class B { public: int id; B() { id = 0; } ~B() { printf("~B\n"); } };
int main(void) { A a; B b; printf("body\n"); return 0; }
