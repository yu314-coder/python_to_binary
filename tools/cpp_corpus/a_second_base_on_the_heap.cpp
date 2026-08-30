#include <cstdio>
static int freed = 0;
struct L { int n; L() : n(1) {} virtual int who() { return 1; } virtual int add(int a, int b) { return a + b; } virtual ~L() {} };
struct R { int m; R() : m(2) {} virtual int who() { return 2; } virtual int scale(int a) { return a * 2; } virtual ~R() { ++freed; } };
struct D : L, R {
    int k;
    D() : k(3) {}
    int who() { return 30 + k; }
    int scale(int a) { return a * 10 + k; }
    ~D() {}
};
int main() {
    D d;
    L *pl = &d; R *pr = &d;
    R *heap = new D();
    int got = heap->scale(1);
    delete heap;
    printf("%d %d %d %d %d %d\n", pl->who(), pr->who(), pr->scale(4), pl->add(1, 2), got, freed);
    return 0;
}
