#include <cstdio>
static char order[32]; static int at = 0;
struct Named {
    const char *name; int tag;
    Named() { name = "?"; tag = 1; order[at++] = 'N'; }
    ~Named() { order[at++] = 'n'; }
    const char *who() const { return name; }
};
struct Counted {
    int n;
    Counted() { n = 0; order[at++] = 'C'; }
    ~Counted() { order[at++] = 'c'; }
    int bump() { return ++n; }
    int count() const { return n; }
};
struct Thing : Named, Counted {
    int extra;
    Thing() : extra(9) { name = "thing"; order[at++] = 'T'; }
    ~Thing() { order[at++] = 't'; }
    int all() { bump(); bump(); return count() + extra + tag; }
};
int main() {
    { Thing t; printf("%s %d %d %d %d\n", t.who(), t.all(), t.extra, t.n, t.tag); }
    order[at] = 0;
    printf("%s\n", order);
    return 0;
}
