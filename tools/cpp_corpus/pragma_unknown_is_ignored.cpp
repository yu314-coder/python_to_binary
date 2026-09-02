#include <cstdio>
#pragma frobnicate 4
#pragma acme vectorize always
#pragma ivdep
#pragma region section
struct Counter {
    int total;
    Counter() : total(0) {}
    void add(int n) { total += n; }
};
#pragma endregion
#define HINT(x) _Pragma(#x)
HINT(acme unroll always)
int main() {
    Counter c;
#pragma unknown_to_everyone
    for (int i = 1; i <= 4; ++i) c.add(i);
    _Pragma("acme hint")
    printf("%d\n", c.total);
    return 0;
}
