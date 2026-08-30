#include <cstdio>
struct R { int n; R() : n(7) {} void bump() { ++n; } };
static R &plain() { static R held; return held; }
struct S { static R &member() { static R kept; return kept; } };
int main() { plain().bump(); printf("%d %d\n", plain().n, S::member().n); return 0; }
