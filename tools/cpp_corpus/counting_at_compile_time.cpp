#include <cstdio>
template <typename T> static T total(T v) { return v; }
template <typename T, typename... R> static T total(T v, R... rest) { return v + total(rest...); }
template <int N> struct Fact { static const int value = N * Fact<N - 1>::value; };
template <> struct Fact<0> { static const int value = 1; };
int main() {
    printf("%d %d %d %d\n", total(1), total(1, 2), total(1, 2, 3, 4), Fact<5>::value);
    return 0;
}
