#include <cstdio>
class V {
public:
    int x, y;
    V() : x(0), y(0) { }
    V(int a, int b) : x(a), y(b) { }
    V operator+(const V &o) const { return V(x + o.x, y + o.y); }
    V operator-(const V &o) const { return V(x - o.x, y - o.y); }
    V &operator+=(const V &o) { x += o.x; y += o.y; return *this; }
    bool operator==(const V &o) const { return x == o.x && y == o.y; }
    bool operator!=(const V &o) const { return !(*this == o); }
    int operator[](int i) const { return i == 0 ? x : y; }
    V operator-() const { return V(-x, -y); }
};
int main() {
    V a(1, 2), b(3, 4);
    V c = a + b; V d = b - a; c += a;
    V e = -a;
    printf("%d %d %d %d %d %d %d %d %d\n", c.x, c.y, d.x, d.y, e.x, e.y,
           c[0], (int)(a == a), (int)(a != b));
    return 0;
}
