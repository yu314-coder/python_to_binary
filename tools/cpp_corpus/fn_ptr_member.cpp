#include <stdio.h>
static int dbl(int n){ return n*2; }
class H { public: int (*op)(int); H(){ op = dbl; } int use(int n){ return op(n); } };
int main(void){ H h; printf("%d\n", h.use(4)); return 0; }
