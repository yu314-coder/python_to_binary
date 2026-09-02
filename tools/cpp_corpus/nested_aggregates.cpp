#include <cstdio>
struct Inner { int a; };
union U1 { Inner in; long l; };
struct S2 { union Named { int x; float f; } named; int tail; };
union U3 { struct Pt { int x, y; } p; long l; };
struct D4 { struct Mid { struct Bot { int v; } b; } m; };
int main() {
    U1 u; u.in.a = 3;
    S2 s; s.named.x = 4; s.tail = 5;
    U3 w; w.p.x = 6;
    D4 d; d.m.b.v = 7;
    printf("%d %d %d %d %d\n", u.in.a, s.named.x, s.tail, w.p.x, d.m.b.v);
    return 0;
}
