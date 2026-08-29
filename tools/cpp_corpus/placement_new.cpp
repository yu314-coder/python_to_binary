#include <cstdio>
#include <new>
struct P { int v; P(int n) : v(n) {} };
int main() { char room[sizeof(P)]; P *p = new (room) P(7); printf("%d\n", p->v); return 0; }
