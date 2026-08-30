#include <cstdio>
static int twice(int v) { return v * 2; }
static int thrice(int v) { return v * 3; }
struct C { int n; C(int v) : n(v) {} int get() const { return n; } int add(int m) const { return n + m; } };
int main() {
    int (*f)(int) = twice;
    int one = f(5);
    f = thrice;
    int (*table[2])(int) = { twice, thrice };
    int (C::*m)() const = &C::get;
    C c(4);
    printf("%d %d %d %d %d\n", one, f(5), table[0](2), table[1](2), (c.*m)());
    return 0;
}
