#include <stdio.h>
class B { public: int n; B(int v) { n = v; } };
int main() { B a(3); B b = a; printf("%d\n", b.n); return 0; }
