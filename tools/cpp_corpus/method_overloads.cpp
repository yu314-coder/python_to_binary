#include <stdio.h>
class Calc {
public:
    int base;
    Calc() { base = 10; }
    Calc(int b) { base = b; }
    int add() { return base; }
    int add(int a) { return base + a; }
    int add(int a, int b) { return base + a + b; }
    int scale(int k) { return base * k; }
};
class Deep : public Calc {
public:
    Deep() { base = 3; }
    int twice(int v) { return add(v) + add(v); }
};
int main(void) {
    Calc c; Calc d(100);
    Deep e;
    printf("%d %d %d %d %d %d\n",
           c.add(), c.add(5), c.add(5, 6), d.add(1), c.scale(4), e.twice(2));
    return 0;
}
