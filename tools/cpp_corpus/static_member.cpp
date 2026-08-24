#include <stdio.h>
class C { public: static int count; int n; C(){ n = ++count; } };
int C::count = 0;
int main(void){ C a; C b; printf("%d %d %d\n", a.n, b.n, C::count); return 0; }
