#include <cstdio>
template <int N> struct Buf { int v[N]; int size() const { return N; } };
template <int N> int total(const Buf<N> &b) { int t = 0; for (int i = 0; i < N; i++) t += b.v[i]; return t; }
int main() {
    Buf<3> a; a.v[0] = 1; a.v[1] = 2; a.v[2] = 3;
    Buf<5> b; for (int i = 0; i < 5; i++) b.v[i] = i;
    printf("%d %d %d %d\n", a.size(), total(a), b.size(), total(b));
    return 0;
}
