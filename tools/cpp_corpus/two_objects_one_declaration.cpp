#include <stdio.h>
class V { public: int n; V(int v):n(v){} V operator+(const V&o) const { return V(n+o.n); } };
int main(){ V a(2), b(3); V c = a + b; printf("%d\n", c.n); return 0; }
