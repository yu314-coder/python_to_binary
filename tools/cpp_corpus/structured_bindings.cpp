#include <cstdio>
struct P { int x; int y; };
int main() { P p{1,2}; auto [a, b] = p; printf("%d %d\n", a, b); return 0; }
