#include <stdio.h>
class B { public: int n; B(){n=0;} B &add(int v){ n += v; return *this; } };
int main(void){ B b; b.add(2).add(3).add(4); printf("%d\n", b.n); return 0; }
