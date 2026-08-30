#include <cstdio>
struct A { protected: int base; public: A(int v) : base(v) {} virtual int val() const { return base; } };
struct B : A { B(int v) : A(v * 2) {} int val() const { return A::val() + 1; } };
struct C : B { C(int v) : B(v + 1) {} int val() const { return B::val() * 10; } int raw() const { return base; } };
int main() {
    C c(2);
    A *p = &c;
    printf("%d %d %d\n", c.val(), p->val(), c.raw());
    return 0;
}
