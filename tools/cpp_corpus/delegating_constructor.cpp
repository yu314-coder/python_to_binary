#include <cstdio>
struct P { int x, y; P(int a, int b) : x(a), y(b) {} P() : P(1, 2) {} int sum() { return x + y; } };
int main() { P a; P b(3, 4); printf("%d %d\n", a.sum(), b.sum()); return 0; }
