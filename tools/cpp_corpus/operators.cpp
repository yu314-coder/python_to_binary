#include <stdio.h>
class V {
public:
    int x;
    V() { x = 0; }
    void set(int n) { x = n; }
    V operator+(V o) { V r; r.set(x + o.x); return r; }
    int operator==(V o) { return x == o.x; }
    int get() { return x; }
};
int main(void) {
    V a; a.set(3);
    V b; b.set(4);
    V c = a + b;
    V d; d.set(7);
    printf("%d %d %d\n", c.get(), c == d, a == b);
    return 0;
}
