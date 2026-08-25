#include <stdio.h>
class V { public: int x, y; V(int a, int b) { x = a; y = b; }
  V plus(V o) { return V(x + o.x, y + o.y); } };
int sum(V a, V b) { return a.x + b.x; }
int main() {
    V p = V(1, 2);
    V q(3, 4);
    V r = p.plus(q);
    printf("%d %d %d\n", r.x, r.y, sum(V(5, 0), V(6, 0)));
    return 0;
}
