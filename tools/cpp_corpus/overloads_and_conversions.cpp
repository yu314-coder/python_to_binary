#include <cstdio>
struct Base { virtual int who() { return 1; } };
struct Derived : Base { int who() { return 2; } };
static int f(double d) { return 10; }
static int f(int i) { return 20; }
static int g(Base *b) { return b->who(); }
static int h(long v) { return (int)v; }
int main() {
    Derived d;
    printf("%d %d %d %d %d\n", f(1), f(1.0), f('a'), g(&d), h(7));
    return 0;
}
