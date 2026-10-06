// A member between two others throws, and later the body throws: each time exactly what was built is taken apart, base last.
#include <cstdio>
#include <string>
struct E { int c; E(int v) : c(v) {} };
struct Part { int id; Part() : id(0) { printf("default part\n"); } Part(int i) : id(i) { if (i < 0) throw E(i); printf("part %d\n", id); } ~Part() { printf("unpart %d\n", id); } };
struct Base { Base() { printf("base\n"); } ~Base() { printf("unbase\n"); } };
struct Whole : Base { Part a; Part b; std::string s; Part c; Whole(int x, int y) : b(x), s("text"), c(3) { printf("whole\n"); if (y < 0) throw E(y); } ~Whole() { printf("unwhole\n"); } };
int main() {
    try { Whole w(-2, 1); printf("not reached\n"); } catch (const E &e) { printf("caught %d\n", e.c); }
    try { Whole w(2, -9); printf("not reached\n"); } catch (const E &e) { printf("caught %d\n", e.c); }
    try { Whole w(2, 1); printf("built\n"); } catch (const E &e) { printf("not reached\n"); }
    return 0;
}
