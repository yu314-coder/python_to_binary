// `throw E{7, 8};` - an aggregate built from braces where it is thrown.
#include <cstdio>
struct E { int c; int d; };
int main() {
    try { throw E{7, 8}; } catch (const E &e) { printf("%d %d\n", e.c, e.d); }
    return 0;
}
