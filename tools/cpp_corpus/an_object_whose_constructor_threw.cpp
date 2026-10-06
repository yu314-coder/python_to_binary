// An object whose constructor threw was never built, and is not taken apart: R's count of live objects ends at zero. It ended at -3, one destructor too many for each object whose constructor had thrown.
#include <cstdio>
static int live = 0;
struct E { int c; E(int v) : c(v) {} };
struct R { int v; R(int x) : v(x) { if (x < 0) throw E(x); ++live; } ~R() { --live; } };
struct Q { int v; Q(int x) { v = x; if (x < 0) throw E(x); ++live; } ~Q() { --live; } };
int take(const R &r) { return r.v; }
int main() {
    int got = 0;
    try { R a(1); R b(-2); printf("not reached\n"); } catch (const E &e) { got += e.c; }
    try { Q a(1); Q b(-3); printf("not reached\n"); } catch (const E &e) { got += e.c; }
    try { int t = take(R(-4)); printf("not reached %d\n", t); } catch (const E &e) { got += e.c; }
    printf("%d %d\n", got, live);
    return 0;
}
