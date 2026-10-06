// `catch (...)` takes an object thrown from braces and a number alike.
#include <cstdio>
struct E { int c; };
int main() {
    int n = 0;
    try { throw E{3}; } catch (...) { n++; }
    try { throw 7; } catch (...) { n += 10; }
    printf("%d\n", n);
    return 0;
}
