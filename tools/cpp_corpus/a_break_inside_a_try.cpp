// A `break` out of a loop from inside a try block.
#include <cstdio>
struct E { int c; };
int main() {
    int i = 0;
    for (; i < 10; i++) {
        try { if (i == 4) break; if (i == 7) throw E{i}; } catch (const E &e) { printf("never\n"); }
    }
    printf("%d\n", i);
    return 0;
}
