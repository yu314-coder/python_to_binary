// The C++ twin of a_typedef_base_under_a_nested_declarator.c: `Op (*p2)(int)
// = pick;` with `Op` a typedef of a pointer to a function, at block scope,
// file scope, as a parameter, an array, a cast and through a second typedef.
// The C++ stage gives `pick` a typedef of its own the moment its name is
// used as a value - `typedef Op (*__py2bin_fn_pick)(int w);` - and wrote it
// at the top of the file, above the typedef that declares `Op`, so the C
// stage refused every one of these programs at a line nobody wrote. It now
// goes right after `Op`. The template call is the reason the typedef exists
// at all: `call(pick)` is a copy of `call` taking that type by name.
#include <cstdio>
typedef int (*Op)(int, int);
typedef Op (*Picker)(int);
static int r_add(int a, int b) { return a + b; }
static int r_mul(int a, int b) { return a * b; }
static Op pick(int w) { return w ? r_mul : r_add; }
template<class F> int call(F f, int w) { return f(w)(5, 6); }
Op (*gp)(int) = pick;
Op (*gq)(int);
Op (*table[2])(int) = { pick, pick };
static int use(Op (*q)(int), int w) { return q(w)(3, 4); }
int main() {
    Op (*p2)(int) = pick;
    Picker P = pick;
    Op o = P(1);
    gq = pick;
    printf("%d %d %d %d\n", p2(0)(7, 8), gp(1)(7, 8), gq(0)(1, 2), o(2, 5));
    printf("%d %d\n", use(pick, 0), use(pick, 1));
    printf("%d %d\n", ((Op (*)(int))pick)(1)(2, 3), ((Picker)pick)(0)(2, 3));
    printf("%d %lu %d\n", table[1](1)(2, 2), (unsigned long)sizeof(Op (*)(int)), (int)(gp == p2));
    printf("%d %d\n", call(pick, 0), call(pick, 1));
    return 0;
}
