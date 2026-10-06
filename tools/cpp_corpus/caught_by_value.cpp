// `catch (E e)` copies what was thrown with E's own copy, though E has no default constructor.
#include <cstdio>
struct E { int c; E(int v) : c(v) {} };
int main() {
    try { throw E(5); } catch (E e) { e.c += 1; printf("%d\n", e.c); }
    return 0;
}
