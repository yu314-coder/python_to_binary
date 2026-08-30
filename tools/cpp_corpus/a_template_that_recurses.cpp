#include <cstdio>
template <int N> struct Chain { static int get() { return N + Chain<N - 1>::get(); } };
template <> struct Chain<0> { static int get() { return 0; } };
int main() { printf("%d\n", Chain<3>::get()); return 0; }
