#include <stdio.h>
class R { public: int a; int b; R(){a=0;b=0;} };
R make(int x){ R r; r.a = x; r.b = x*2; return r; }
int main(void){ R r = make(4); printf("%d %d\n", r.a, r.b); return 0; }
