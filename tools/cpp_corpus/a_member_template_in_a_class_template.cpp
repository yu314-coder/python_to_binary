#include <cstdio>
template <typename T, int N>
struct Fixed {
    T slot[N];
    int used;
    Fixed() : used(0) {}
    void add(T v) { if (used < N) slot[used++] = v; }
    template <typename U> U sumAs() const { U t = 0; for (int i = 0; i < used; ++i) t = t + (U)slot[i]; return t; }
    int room() const { return N - used; }
};
int main() {
    Fixed<int, 4> f;
    f.add(1); f.add(2); f.add(3);
    Fixed<double, 2> g;
    g.add(1.5); g.add(2.5);
    printf("%d %d %g %d\n", f.sumAs<int>(), f.room(), g.sumAs<double>(), g.room());
    return 0;
}
