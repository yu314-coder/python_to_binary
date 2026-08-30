#include <cstdio>
struct Base { int v; Base(int x) : v(x) {} virtual int get() { return v; } };
struct Only : virtual Base { Only() : Base(5) {} int twice() { return v * 2; } };
int main() { Only o; Base *p = &o; printf("%d %d %d\n", o.v, o.twice(), p->get()); return 0; }
