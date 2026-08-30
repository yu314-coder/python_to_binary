#include <cstdio>
static int order[8]; static int at = 0;
struct B { virtual ~B() { order[at++] = 1; } };
struct D : B { ~D() { order[at++] = 2; } };
int main() {
    B *p = new D();
    delete p;
    { D d; }
    printf("%d %d %d %d %d\n", at, order[0], order[1], order[2], order[3]);
    return 0;
}
