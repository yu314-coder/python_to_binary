// A method taking `Shape *const &`, handed a named pointer, an address and a new object.
#include <cstdio>
struct Shape { int n; };
struct Bag { Shape *items[4]; int count; Bag() : count(0) {}
    void add(Shape *const &p) { items[count++] = p; } };
int main() { Shape s{3}; Shape t{4}; Bag b; Shape *p = &s;
    b.add(p); b.add(&t); b.add(new Shape());
    printf("%d %d %d %d\n", b.count, b.items[0]->n, b.items[1]->n, b.items[2]->n); return 0; }
