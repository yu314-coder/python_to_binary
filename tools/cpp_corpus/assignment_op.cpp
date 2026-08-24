#include <stdio.h>
class C { public: int n; C(){n=1;} C &operator=(const C &o){ n = o.n + 5; return *this; } };
int main(void){ C a; C b; a.n = 3; b = a; printf("%d\n", b.n); return 0; }
