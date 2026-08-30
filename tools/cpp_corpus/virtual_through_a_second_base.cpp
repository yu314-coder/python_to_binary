#include <cstdio>
struct L { virtual int l() { return 1; } virtual ~L() {} };
struct R { virtual int r() { return 2; } virtual ~R() {} };
struct D : L, R { int l() { return 10; } int r() { return 20; } };
int main() {
    D d;
    L *pl = &d; R *pr = &d;
    printf("%d %d %d\n", pl->l(), pr->r(), d.l() + d.r());
    return 0;
}
