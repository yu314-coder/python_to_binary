#include <cstdio>
struct Inner { int v; Inner() : v(0) {} };
struct Outer {
    Inner one;
    Inner &pick() { return one; }
    const Inner &look() const { return one; }
};
int main() {
    Outer o;
    o.pick().v = 5;
    Inner &r = o.pick();
    r.v += 2;
    printf("%d %d %d\n", o.one.v, o.pick().v, o.look().v);
    return 0;
}
