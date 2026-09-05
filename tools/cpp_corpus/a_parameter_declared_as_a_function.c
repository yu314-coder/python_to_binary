/* A parameter written as a function, `int f(int)`, which C adjusts to a
   pointer to that function (C11 6.7.6.3p8) - so `int apply(int f(int), int
   x)` and `int apply(int (*f)(int), int x)` are one prototype, and `f(x)`
   inside is a call through the pointer. py2bin's definition parser adjusted
   an array parameter and not a function one: `apply(twice, 4)` was refused
   for handing a pointer to an `int (int)`, and the two spellings of the
   prototype were refused as disagreeing. */
#include <stdio.h>
static int twice(int x) { return 2 * x; }
static int square(int x) { return x * x; }
int apply(int f(int), int x);
int apply(int (*f)(int), int x) { return f(x); }
static int both(int g(int), int h(int), int v) { return g(v) + h(v); }
static int pick_and_apply(int which, int a(int), int b(int), int v) {
    return apply(which ? a : b, v);
}
int main(void) {
    printf("%d %d %d %d\n", apply(twice, 4), both(twice, square, 5),
           pick_and_apply(1, square, twice, 6), pick_and_apply(0, square, twice, 6));
    return 0;
}
