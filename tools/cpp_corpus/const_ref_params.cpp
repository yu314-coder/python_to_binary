#include <stdio.h>
class Big { public: int n; Big(){n=7;} int get() const { return n; } };
int read(const Big &b) { return b.get(); }
void bump(Big &b) { b.n = b.n + 1; }
int main(void){ Big x; bump(x); printf("%d %d\n", read(x), x.n); return 0; }
