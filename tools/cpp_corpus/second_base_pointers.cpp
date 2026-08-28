#include <cstdio>
struct Named { const char *name; Named() { name = "?"; } const char *who() const { return name; } };
struct Counted { int n; Counted() { n = 0; } int bump() { return ++n; } int count() const { return n; } };
struct Thing : Named, Counted {
    int extra;
    Thing() : extra(9) { name = "thing"; }
    int all() { bump(); bump(); return count() + extra; }
};
static int through(Counted *c) { return c->bump(); }
int main() {
    Thing t;
    printf("%s %d %d\n", t.who(), t.all(), t.extra);
    printf("%d %d\n", through(&t), t.n);
    Named *n = &t;
    Counted *c = &t;
    printf("%s %d\n", n->who(), c->count());
    return 0;
}
