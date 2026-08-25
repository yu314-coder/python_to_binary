#include <stdio.h>
class V { public: int n; V(int x) { n = x; } };
int main() { V t = V(5); printf("%d\n", t.n); return 0; }
