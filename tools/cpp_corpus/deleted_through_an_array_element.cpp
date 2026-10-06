// `delete all[0]` runs the destructor of what the element points at, through the virtual one.
#include <cstdio>
struct Shape { virtual ~Shape() { printf("~Shape\n"); } virtual int area() const = 0; };
struct Sq : Shape { int s; Sq(int v) : s(v) {} ~Sq() { printf("~Sq %d\n", s); } int area() const override { return s * s; } };
int main() { Shape *all[2] = {new Sq(5), new Sq(6)}; delete all[0]; Shape *p = all[1]; delete p; return 0; }
