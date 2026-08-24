#include <stdio.h>
class Box { public: int v; Box() { v = 0; } void set(int x) { v = x; } int get() { return v; } };
void fill(Box *b, int x) { b->set(x); }
int main(void) { Box b; fill(&b, 42); printf("%d\n", b.get()); return 0; }
