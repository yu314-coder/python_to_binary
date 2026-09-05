/* A pointer to a function whose result is itself a typedef'd pointer to a
   function: `Op (*p2)(int) = pick;`, with `Op` a typedef and the declarator
   nested around the new name. The same shape at file scope, as a parameter,
   as an array, as the type a cast names, and through a second typedef
   (`typedef Op (*Picker)(int);`). Every value here reaches the same two
   functions, so what is printed says which path each declaration took. */
#include <stdio.h>
typedef int (*Op)(int, int);
typedef Op (*Picker)(int);
static int r_add(int a, int b) { return a + b; }
static int r_mul(int a, int b) { return a * b; }
static Op pick(int w) { return w ? r_mul : r_add; }
Op (*gp)(int) = pick;
Op (*gq)(int);
Op (*table[2])(int) = { pick, pick };
static int use(Op (*q)(int), int w) { return q(w)(3, 4); }
int main(void) {
    Op (*p2)(int) = pick;
    Picker P = pick;
    Op o = P(1);
    gq = pick;
    printf("%d %d %d %d\n", p2(0)(7, 8), gp(1)(7, 8), gq(0)(1, 2), o(2, 5));
    printf("%d %d\n", use(pick, 0), use(pick, 1));
    printf("%d %d\n", ((Op (*)(int))pick)(1)(2, 3), ((Picker)pick)(0)(2, 3));
    printf("%d %lu %d\n", table[1](1)(2, 2), (unsigned long)sizeof(Op (*)(int)), (int)(gp == p2));
    return 0;
}
