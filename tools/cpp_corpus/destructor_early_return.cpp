#include <stdio.h>
int alive = 0;
class Res { public: Res() { alive = alive + 1; } ~Res() { alive = alive - 1; } };
int f(int n) { Res r; if (n > 0) { return 1; } return 2; }
int main(void) { int v = f(1); printf("%d %d\n", v, alive); return 0; }
