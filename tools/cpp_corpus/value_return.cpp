#include <stdio.h>
class V {
public:
    int x;
    V() { x = 0; }
    void set(int n) { x = n; }
    V plus(V o) { V r; r.set(x + o.x); return r; }
    int get() { return x; }
};
int main(void) {
    V a; a.set(3);
    V b; b.set(4);
    V c = a.plus(b);
    printf("%d %d %d\n", a.get(), b.get(), c.get());
    return 0;
}
