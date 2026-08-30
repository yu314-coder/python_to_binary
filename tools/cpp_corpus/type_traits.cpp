#include <cstdio>
#include <type_traits>
template <typename T> struct Kind { static int id() { return 0; } };
template <typename T> static int sizeOf() { return (int)sizeof(T); }
int main() {
    printf("%d %d %d %d\n", (int)std::is_same<int, int>::value,
           (int)std::is_same<int, double>::value, sizeOf<int>(), sizeOf<double>());
    return 0;
}
