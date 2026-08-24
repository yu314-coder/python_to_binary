#include <stdio.h>
class Z { public: int v; int w; Z() { v = 2; w = twice(); } int twice() { return v * 2; } };
int main(void) { Z z; printf("%d %d\n", z.v, z.w); return 0; }
