#include <stdio.h>
class A { public: virtual int id() { return 1; } virtual int mix() { return id() * 10; } };
class B : public A { public: int id() { return 2; } };
class C : public B { public: int id() { return 3; } int mix() { return id() * 100; } };
class D : public C { public: };
int main(void) {
    A a; B b; C c; D d;
    A *all[4]; all[0] = &a; all[1] = &b; all[2] = &c; all[3] = &d;
    for (int i = 0; i < 4; i++) printf("%d/%d ", all[i]->id(), all[i]->mix());
    printf("\n");
    return 0;
}
