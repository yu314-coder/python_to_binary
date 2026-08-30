#include <cstdio>
struct V {
    int x;
    V(int v) : x(v) {}
    bool operator==(const V &o) const { return x == o.x; }
    bool operator<(const V &o) const { return x < o.x; }
    V &operator++() { ++x; return *this; }
    V operator++(int) { V old(x); ++x; return old; }
    operator bool() const { return x != 0; }
};
int main() {
    V a(3), b(3), c(5);
    V d(1); ++d; V e = d++;
    printf("%d %d %d %d %d %d\n", a == b, a == c, a < c, d.x, e.x, (bool)V(0) ? 1 : 0);
    return 0;
}
