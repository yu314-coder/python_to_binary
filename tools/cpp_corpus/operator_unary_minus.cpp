#include <stdio.h>
class V { public: int n; V(int v):n(v){} V operator-() const { return V(-n); } };
int main(){ V a(5); V b = -a; printf("%d\n", b.n); return 0; }
