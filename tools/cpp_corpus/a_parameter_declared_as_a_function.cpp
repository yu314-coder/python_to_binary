// The C++ twin of a_parameter_declared_as_a_function.c: a parameter spelled
// as a function, `int f(int)`, is a pointer to that function, and a call on
// it inside the body is a call through the pointer. The C++ stage hands the
// parameter through as written, so the C stage's missing adjustment refused
// the definition here exactly as it did in C.
#include <stdio.h>
static int twice(int x) { return 2 * x; }
static int square(int x) { return x * x; }
int apply(int f(int), int x);
int apply(int (*f)(int), int x) { return f(x); }
static int both(int g(int), int h(int), int v) { return g(v) + h(v); }
static int pick_and_apply(int which, int a(int), int b(int), int v) {
    return apply(which ? a : b, v);
}
int main() {
    printf("%d %d %d %d\n", apply(twice, 4), both(twice, square, 5),
           pick_and_apply(1, square, twice, 6), pick_and_apply(0, square, twice, 6));
    return 0;
}
