// An inner try whose one handler is for another type lets the exception through to the outer handler.
#include <cstdio>
struct A { int v; A(int x) : v(x) {} };
struct B { int v; B(int x) : v(x) {} };
int main() {
    int trace = 0;
    try {
        try { throw B(4); }
        catch (const A &a) { trace += 1; }
        trace += 10;
    } catch (const B &b) { trace += 100 * b.v; }
    printf("%d\n", trace);
    return 0;
}
