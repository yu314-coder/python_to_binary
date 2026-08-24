#include <stdio.h>
class E { public: int n; explicit E(int v) { n = v; } inline int get(){ return n; } };
int main(void){ E e(9); printf("%d\n", e.get()); return 0; }
