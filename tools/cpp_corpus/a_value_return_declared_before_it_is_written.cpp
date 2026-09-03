/* A function answering an object by value, declared above and defined below.
   The definition becomes a hidden pointer the caller provides; the prototype
   has to say the same thing, and one spelled with `static` or with the
   `struct` keyword was passed over - so the two disagreed about what the
   function answers. `make(void)` is the other half: the word means no
   parameters, and the hidden pointer written in front of it left a parameter
   of type void behind. */
#include <cstdio>
struct Point { int x; int y; };
static struct Point make(void);
static Point add(Point a, Point b);

int main() {
    Point m = make();
    Point c = add(m, make());
    printf("%d %d %d %d\n", m.x, m.y, c.x, c.y);
    return 0;
}

static struct Point make(void) { Point r; r.x = 3; r.y = 4; return r; }
static Point add(Point a, Point b) {
    Point r; r.x = a.x + b.x; r.y = a.y + b.y; return r;
}
