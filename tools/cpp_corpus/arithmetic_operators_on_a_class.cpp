#include <cstdio>
struct V {
    int x;
    V() : x(0) {}
    V(int v) : x(v) {}
    V operator+(const V &o) const { return V(x + o.x); }
    V &operator+=(const V &o) { x += o.x; return *this; }
    V operator-() const { return V(-x); }
};
int main() {
    V a(1), b(2), c(3);
    V d = a + b + c;
    V e = -d;
    V f(10); f += b;
    printf("%d %d %d\n", d.x, e.x, f.x);
    return 0;
}
