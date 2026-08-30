#include <cstdio>
struct R { int n; R() : n(7) {} void bump() { ++n; } };
static R &plain() { static R held; return held; }
int main() { plain().bump(); printf("%d\n", plain().n); return 0; }
