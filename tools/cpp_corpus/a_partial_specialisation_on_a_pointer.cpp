#include <cstdio>
#include <type_traits>
template <typename T> struct IsPointer { static const int value = 0; };
template <typename T> struct IsPointer<T *> { static const int value = 1; };
template <typename T> static int rank(T v) { return IsPointer<T>::value ? 10 : 20; }
int main() {
    int n = 5;
    printf("%d %d %d %d\n", IsPointer<int>::value, IsPointer<int *>::value, rank(n), rank(&n));
    return 0;
}
