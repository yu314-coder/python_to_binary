#include <stdio.h>
class C { public: int n; C(int v):n(v){} C &operator++(){ n++; return *this; } };
int main(){ C c(1); ++c; ++c; printf("%d\n", c.n); return 0; }
