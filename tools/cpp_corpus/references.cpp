#include <stdio.h>
class Box {
public:
    int v;
    Box() { v = 1; }
    int get() { return v; }
    void take(Box &other) { v = v + other.v; }
    void add(int &slot, int by) { slot = slot + by; }
};
void bump(Box &b, int by) { b.v = b.v + by; }
int read(const Box &b) { return b.v * 2; }
void swap(int &a, int &b) { int t = a; a = b; b = t; }
int main(void) {
    Box a; Box c;
    bump(a, 9);
    c.v = 100;
    a.take(c);
    int &alias = a.v;
    alias = alias + 1;
    int x = 3, y = 8;
    swap(x, y);
    int counter = 0;
    a.add(counter, 5);
    printf("%d %d %d %d %d\n", a.get(), read(a), x, y, counter);
    return 0;
}
