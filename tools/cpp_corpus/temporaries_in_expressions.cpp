#include <cstdio>
static int copies = 0;
struct V {
    int x;
    V(int v) : x(v) {}
    V(const V &o) : x(o.x) { ++copies; }
    V add(const V &o) const { return V(x + o.x); }
};
static V make(int n) { return V(n); }
static int read(const V &v) { return v.x; }
int main() {
    V a(1);
    int one = read(make(5));
    int two = a.add(V(2)).x;
    V three = make(3).add(a);
    printf("%d %d %d\n", one, two, three.x);
    return 0;
}
