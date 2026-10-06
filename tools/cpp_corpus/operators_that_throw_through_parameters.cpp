// A throwing subscript and a throwing functor reached through reference parameters, and a throwing lambda handed to a template.
#include <cstdio>
#include <vector>
struct E { int c; E(int v) : c(v) {} };
struct Checked { int d[3]; int &operator[](int i) { if (i < 0 || i >= 3) throw E(i); return d[i]; } };
struct Limit { int most; int operator()(int x) const { if (x > most) throw E(x); return x * 2; } };
int sum_of(Checked &c, int n) { int s = 0; for (int i = 0; i < n; i++) { s += c[i]; } return s; }
int apply(const Limit &f, int x) { return f(x) + 1; }
template <typename F> void each(std::vector<int> &v, F f) { for (int x : v) f(x); }
int main() {
    Checked c; c.d[0] = 1; c.d[1] = 2; c.d[2] = 3;
    Limit lim{10};
    int got = 0;
    try { got += sum_of(c, 3); got += apply(lim, 4); got += sum_of(c, 5); } catch (const E &e) { got += e.c * 100; }
    try { got += apply(lim, 50); } catch (const E &e) { got += e.c * 10000; }
    std::vector<int> v = {1, 2, 30, 4};
    int seen = 0;
    try { each(v, [&seen](int x) { if (x > 10) throw E(x); seen += x; }); } catch (const E &e) { got += e.c; }
    printf("%d %d\n", got, seen);
    return 0;
}
