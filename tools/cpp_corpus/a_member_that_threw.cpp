// A member whose constructor threw: the members before it are taken apart, the rest are never built, and the whole object is not destroyed.
#include <cstdio>
struct E { int c; E(int v) : c(v) {} };
struct Part { int id; Part(int i) : id(i) { if (i < 0) throw E(i); printf("part %d\n", id); } ~Part() { printf("unpart %d\n", id); } };
struct Whole { Part a; Part b; Part c; Whole(int x) : a(1), b(x), c(3) { printf("whole\n"); } ~Whole() { printf("unwhole\n"); } };
int main() {
    try { Whole w(-2); printf("not reached\n"); } catch (const E &e) { printf("caught %d\n", e.c); }
    try { Whole w(2); } catch (const E &e) { printf("not reached\n"); }
    return 0;
}
