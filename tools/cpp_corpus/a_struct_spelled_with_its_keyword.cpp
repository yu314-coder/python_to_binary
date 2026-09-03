/* `struct Point p` and `const Point p` name a Point as surely as `Point p`
   does, and a C++ program is free to write either. Read as though a type
   were always one word, neither reached the pass that turns a by-value
   object into the pointer it travels as. */
#include <cstdio>
struct Point { int x; int y; };
static int sum(const Point p) { return p.x + p.y; }
static struct Point add(struct Point a, const struct Point b) {
    struct Point r; r.x = a.x + b.x; r.y = a.y + b.y; return r;
}
int main() {
    struct Point a; a.x = 1; a.y = 2;
    Point b; b.x = 30; b.y = 40;
    struct Point c = add(a, b);
    struct Point d = c;
    d.x = 0;
    printf("%d %d %d %d %d\n", c.x, c.y, d.x, sum(c), sum(b));
    return 0;
}
