#include <cstdio>
struct Base { int a; int b; };
struct Derived : Base { int c; };
int main() {
    Derived d;
    d.a = 1; d.b = 2; d.c = 3;
    Base *p = &d;
    printf("%d %d %d %d %d\n", d.a, d.b, d.c, p->a, (int)sizeof(Derived));
    return 0;
}
