#include <cstdio>
struct B { virtual int t() { return 1; } virtual ~B() {} };
struct D : B { int t() { return 2; } };
int main() {
    double x = 3.9;
    int i = static_cast<int>(x);
    const int c = 5;
    int *p = const_cast<int *>(&c);
    B *b = new D();
    D *d = static_cast<D *>(b);
    printf("%d %d %d %d\n", i, *p, b->t(), d->t());
    delete b;
    return 0;
}
