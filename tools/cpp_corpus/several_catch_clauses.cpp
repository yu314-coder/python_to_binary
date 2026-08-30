#include <cstdio>
struct A { int c; A(int v) : c(v) {} };
struct B { int d; B(int v) : d(v) {} };
static int pick(int n) { if (n == 1) throw A(11); if (n == 2) throw B(22); return n; }
int main() {
    int a = 0, b = 0, c = 0;
    try { pick(1); } catch (const A &e) { a = e.c; } catch (...) { a = -1; }
    try { pick(2); } catch (const A &e) { b = -1; } catch (...) { b = 99; }
    c = pick(3);
    printf("%d %d %d\n", a, b, c);
    return 0;
}
