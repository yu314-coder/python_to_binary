#include <cstdio>
struct Acc {
    int v;
    Acc(int n) : v(n) {}
    Acc &operator+=(int n) { v += n; return *this; }
    Acc &operator++() { v++; return *this; }
    int operator[](int i) const { return v + i; }
    operator int() const { return v; }
};
int main() {
    Acc a(1);
    a += 5; ++a;
    int n = 3; n *= 2; n <<= 1; n -= 1;
    printf("%d %d %d %d\n", a.v, a[2], (int)a, n);
    return 0;
}
