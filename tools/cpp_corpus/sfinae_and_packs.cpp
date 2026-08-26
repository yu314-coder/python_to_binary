#include <stdio.h>
#include <type_traits>

/* SFINAE picking between two, with the answers coming out of the shipped
   <type_traits>. */
template <class T>
typename std::enable_if<std::is_pointer<T>::value, int>::type kind(T v) { return 1; }
template <class T>
typename std::enable_if<std::is_integral<T>::value, int>::type kind(T v) { return 2; }
template <class T>
typename std::enable_if<std::is_floating_point<T>::value, int>::type kind(T v) { return 3; }

/* And a variadic pack, counted and summed. */
static int total() { return 0; }
template <class... Rest> static int total(int a, Rest... rest) { return a + total(rest...); }

template <class... Ts> struct arity { static const int value = sizeof...(Ts); };

int main() {
    int n = 5;
    double d = 1.5;
    printf("%d %d %d\n", kind(&n), kind(n), kind(d));
    printf("%d %d %d\n", total(), total(1, 2), total(1, 2, 3, 4, 5));
    printf("%d %d %d\n", arity<>::value, arity<int>::value,
           arity<int, char, double, long>::value);
    return 0;
}
