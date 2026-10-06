// A lambda that throws, called through the variable that holds it; the handler is outside.
#include <cstdio>
struct E { int c; E(int v) : c(v) {} };
int main() {
    auto check = [](int x) { if (x > 5) throw E(x); return x; };
    int got = 0;
    try { got += check(3); got += check(9); } catch (const E &e) { got += e.c * 10; }
    printf("%d\n", got);
    return 0;
}
