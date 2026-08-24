#include <stdio.h>
class C { public: int n; C(){n=4;} int get() const { return n; } };
int main(void){ C c; printf("%d\n", c.get()); return 0; }
