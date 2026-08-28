#include <cstdio>
constexpr int fact(int n) { return n <= 1 ? 1 : n * fact(n - 1); }
int main() { int a[fact(4)]; printf("%d %d\n", fact(5), (int)(sizeof(a)/sizeof(a[0]))); return 0; }
