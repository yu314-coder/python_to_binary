/* <type_traits>: three questions answered at compile time. */
#include <type_traits>
#include <cstdio>

int main() {
    printf("%d %d\n", (int)std::is_same<int, int>::value,
           (int)std::is_same<int, double>::value);
    printf("%d %d\n", (int)std::is_pointer<int *>::value,
           (int)std::is_pointer<int>::value);
    printf("%d %d\n", (int)std::is_integral<int>::value,
           (int)std::is_integral<double>::value);
    return 0;
}
