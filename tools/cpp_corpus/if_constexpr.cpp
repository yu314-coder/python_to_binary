#include <cstdio>
#include <type_traits>
template <typename T> int kind(T v) {
    if constexpr (sizeof(T) == 4) { return 4; } else { return 8; }
}
template <typename T> int shape(T v) {
    if constexpr (std::is_pointer<T>::value) { return 1; } else { return 0; }
}
constexpr int twice(int n) { return n * 2; }
int main() {
    int n = 3;
    printf("%d %d %d %d %d\n", kind(1), kind(1.0), shape(&n), shape(n), twice(5));
    return 0;
}
