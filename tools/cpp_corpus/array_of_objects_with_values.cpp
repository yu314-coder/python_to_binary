#include <stdio.h>
class A { public: int n; A(int v):n(v){} };
int main(){ A xs[3] = {A(1), A(2), A(3)}; printf("%d%d%d\n", xs[0].n, xs[1].n, xs[2].n); return 0; }
