#include <stdio.h>
class Box { public: int v; Box() { v = 1; } int get() { return v; } };
void bump(Box &b, int by) { b.v = b.v + by; }
int read(const Box &b) { return b.v * 2; }
int main(void) {
    Box a;
    bump(a, 9);
    int &alias = a.v;
    alias = alias + 1;
    printf("%d %d\n", a.get(), read(a));
    return 0;
}
