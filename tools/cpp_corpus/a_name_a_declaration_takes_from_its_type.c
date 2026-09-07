/* C says a declaration takes the name away from whatever had it. `struct
   path path;` and from there on `path` is the variable, not the typedef -
   which is what lets a program keep a `std::filesystem::path` in something
   called `path`, and their Windows companion does exactly that. py2bin read
   the statement after it as another declaration, whose type was `path`, and
   stopped at the `=`. It goes back to being a type on the way out of the
   block, and a `for` clause is a block of its own. */
#include <stdio.h>

typedef struct point { int x; int y; } point;
typedef int count;
typedef int step;

static int total(point p) { return p.x + p.y; }

/* A parameter takes the name from its type too, and for the whole body. */
static int through(point point) {
    point.x = point.x + 1;
    return point.x + point.y;
}

int main(void) {
    point point;
    point.x = 3;
    point.y = 4;
    count count = 2;
    count = count + 5;
    {
        /* A block of its own: inside it the name is still the variable's,
           and the type is reached by its tag. */
        struct point inner;
        inner.x = point.x * 2;
        inner.y = 0;
        printf("%d ", total(inner));
    }
    int walked = 0;
    for (step step = 0; step < 3; step++) { walked += step; }
    /* And here `step` is a type again, because the `for` closed. */
    step after = 9;
    struct point another = {9, 1};
    printf("%d %d %d %d %d %d\n", total(point), count, through(point),
           total(another), walked, after);
    return 0;
}
