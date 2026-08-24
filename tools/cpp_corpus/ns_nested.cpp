#include <stdio.h>
namespace outer {
    namespace inner {
        class Deep { public: int v; Deep() { v = 42; } int get() { return v; } };
        int helper(int n) { return n + 1; }
    }
    int wrap(int n) { return inner::helper(n) * 2; }
}
int main(void) {
    outer::inner::Deep d;
    printf("%d %d\n", d.get(), outer::wrap(3));
    return 0;
}
