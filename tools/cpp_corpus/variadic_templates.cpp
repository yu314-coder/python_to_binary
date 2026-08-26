/* A pack stands for however many arguments are left, including none. Counted
   with sizeof..., written out one per argument, and walked by recursion -
   which stops on the ordinary function of the same name, because C++ prefers
   one of those to a copy of a template. */
#include <stdio.h>

template <class... Ts> struct arity { static const int value = sizeof...(Ts); };

static int total() { return 0; }
template <class... Rest> static int total(int a, Rest... rest) {
    return a + total(rest...);
}

static int widest(int a) { return a; }
template <class... Rest> static int widest(int a, Rest... rest) {
    int other = widest(rest...);
    return a > other ? a : other;
}

int main() {
    printf("%d %d %d %d\n", arity<>::value, arity<int>::value,
           arity<int, char>::value, arity<int, char, double, long>::value);
    printf("%d %d %d %d\n", total(), total(7), total(1, 2), total(1, 2, 3, 4, 5));
    printf("%d %d %d\n", widest(4), widest(4, 9), widest(4, 9, 2, 7));
    return 0;
}
