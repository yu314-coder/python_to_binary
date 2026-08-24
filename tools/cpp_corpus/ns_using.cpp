#include <stdio.h>
namespace util {
    class Counter { public: int n; Counter() { n = 0; } void bump() { n = n + 1; } int get() { return n; } };
    int square(int v) { return v * v; }
}
using namespace util;
int main(void) { Counter c; c.bump(); c.bump(); printf("%d %d\n", c.get(), square(6)); return 0; }
