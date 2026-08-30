#include <cstdio>
struct S {
    int *p; int n;
    S(int v) { n = v; p = new int(v); }
    S(const S &o) { n = o.n; p = new int(*o.p); }
    S &operator=(const S &o) { if (this != &o) { *p = *o.p; n = o.n; } return *this; }
    ~S() { delete p; }
};
int main() {
    S a(1), b(2);
    a = b;
    a = a;
    S c = a;
    c.n = 9;
    printf("%d %d %d %d\n", a.n, *a.p, b.n, c.n);
    return 0;
}
