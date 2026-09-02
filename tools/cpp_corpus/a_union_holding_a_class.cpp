#include <cstdio>
struct Inner { int a; };
union U1 { Inner in; long l; };
int main() { U1 u; u.in.a = 3; printf("%d\n", u.in.a); return 0; }
