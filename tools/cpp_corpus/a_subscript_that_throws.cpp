// An `operator[]` that checks its index and throws, used in a sum.
#include <cstdio>
struct E { int c; E(int v) : c(v) {} };
struct Arr { int d[4]; int &operator[](int i) { if (i < 0 || i >= 4) throw E(i); return d[i]; } };
int main() {
    Arr a; for (int i = 0; i < 4; i++) a.d[i] = i + 1;
    int s = 0;
    try { s += a[1]; s += a[3]; s += a[7]; } catch (const E &e) { s += e.c * 100; }
    printf("%d\n", s);
    return 0;
}
