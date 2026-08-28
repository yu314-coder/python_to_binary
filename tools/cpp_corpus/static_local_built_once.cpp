#include <cstdio>
struct Counter { int n; Counter() : n(0) {} int bump() { return ++n; } };
static int next() { static Counter c; return c.bump(); }
static int lazy() { static int seeded = 41 + 1; return seeded; }
int main() { printf("%d %d %d %d\n", next(), next(), next(), lazy()); return 0; }
