#include <stdio.h>
class W { public: int v; W() { v = 8; } int get() { return v; } };
static int helper(W *w) { return w->get() * 2; }
int main(void) { W w; printf("%d\n", helper(&w)); return 0; }
