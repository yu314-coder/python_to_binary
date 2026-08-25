#include <stdio.h>
class Box { public: int v; Box(int x):v(x){} };
class Holder { public: Box b; Holder():b(3){} };
int main(){ Holder h; printf("%d\n", h.b.v); return 0; }
