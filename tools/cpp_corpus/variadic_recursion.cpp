#include <cstdio>
template <typename T> T total(T v) { return v; }
template <typename T, typename... R> T total(T v, R... rest) { return v + total(rest...); }
int main() { printf("%d\n", total(1, 2, 3, 4)); return 0; }
