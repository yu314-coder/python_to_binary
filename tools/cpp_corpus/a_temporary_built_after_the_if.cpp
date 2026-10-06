// A method building its answer after an `if` that changes what it is built from: the temporary is built there, not at the top.
#include <cstdio>
struct Name { int n; Name(int v) : n(v) { printf("make %d\n", v); } };
struct A { int base; Name make(int k) { if (k < 0) { k = 0; } return Name(k); } };
struct B { int base; Name make(int k) { if (k < 0) { k = 0; } return Name(base + k); } };
struct C { int base; Name make(int k) { if (k < 0) k = 0; return Name(base + k); } };
struct D { int base; Name make(int k) { int j = k; if (j < 0) { j = 0; } Name made(base + j); return made; } };
struct F { int base; int make(int k) { if (k < 0) { k = 0; } return Name(base + k).n; } };
int main() {
    A a; a.base = 100; B b; b.base = 100; C c; c.base = 100; D d; d.base = 100; F f; f.base = 100;
    Name x1 = a.make(-5); Name x2 = b.make(-5); Name x3 = c.make(-5); Name x4 = d.make(-5);
    printf("%d %d %d %d %d\n", x1.n, x2.n, x3.n, x4.n, f.make(-5));
    return 0;
}
