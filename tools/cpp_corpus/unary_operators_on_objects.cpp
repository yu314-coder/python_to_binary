#include <cstdio>
struct V {
    int n;
    V(int v) : n(v) {}
    V operator-() const { return V(-n); }
    V &operator++() { n++; return *this; }
    int operator()(int k) const { return n * k; }
    bool operator!() const { return n == 0; }
};
int main() { V a(3); ++a; V b = -a; printf("%d %d %d %d\n", a.n, b.n, a(5), (int)!b); return 0; }
