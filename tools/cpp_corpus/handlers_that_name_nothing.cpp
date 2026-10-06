// `catch (const E &)` takes an E without naming it, and a second handler takes what the first does not.
#include <cstdio>
struct E { int c; };
struct F { int d; };
int main() {
    int n = 0;
    try { throw F{2}; } catch (const E &) { n = 1; } catch (const F &) { n = 2; }
    printf("%d\n", n);
    return 0;
}
