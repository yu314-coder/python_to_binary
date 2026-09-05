// The C++ twin of a_function_returning_a_function_pointer.c: a function
// whose result is a pointer to a function, spelled without a typedef -
// `int (*pick(int which))(int)`. The C++ stage hands the declarator through
// unchanged, so the C stage's refusal of the shape stopped a C++ program that
// included <openssl/evp.h> exactly as it stopped a C one. A prototype, then
// the definition, and the result called, dereferenced, stored and passed on.
#include <stdio.h>
static int twice(int x) { return 2 * x; }
static int thrice(int x) { return 3 * x; }
int (*pick(int which))(int);
int (*pick(int which))(int) { return which ? thrice : twice; }
int (*kept)(int);
int main() {
    int (*f)(int) = pick(1);
    kept = pick(0);
    printf("%d %d %d %d %d\n", pick(0)(5), (*pick(1))(5), f(7), kept(f(1)), pick(kept(0))(4));
    return 0;
}
