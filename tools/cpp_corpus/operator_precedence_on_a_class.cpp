#include <cstdio>
struct N {
    int x;
    N() : x(0) {}
    N(int v) : x(v) {}
    N operator+(const N &o) const { return N(x + o.x); }
    N operator*(const N &o) const { return N(x * o.x); }
    N operator-(const N &o) const { return N(x - o.x); }
};
int main() {
    N a(2), b(3), c(4);
    N r1 = a + b * c;
    N r2 = (a + b) * c;
    N r3 = a - b - c;
    printf("%d %d %d\n", r1.x, r2.x, r3.x);
    return 0;
}
