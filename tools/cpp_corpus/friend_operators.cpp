#include <cstdio>
struct V {
    int x;
    V(int n) : x(n) {}
    friend V operator+(const V &a, const V &b) { return V(a.x + b.x); }
    friend bool operator<(const V &a, const V &b) { return a.x < b.x; }
};
int main() {
    V a(3), b(4);
    V c = a + b;
    printf("%d %d\n", c.x, (int)(a < b));
    return 0;
}
