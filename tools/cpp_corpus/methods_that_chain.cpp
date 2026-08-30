#include <cstdio>
struct Build {
    int n;
    Build() : n(0) {}
    Build &add(int v) { n += v; return *this; }
    Build &times(int v) { n *= v; return *this; }
    int done() const { return n; }
};
int main() {
    Build b;
    int r = b.add(2).times(3).add(4).done();
    Build c;
    c.add(1).add(2);
    printf("%d %d\n", r, c.done());
    return 0;
}
