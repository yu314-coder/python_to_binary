/* An anonymous struct or union member's members are the enclosing one's.
   C11 says so, and the Windows SDK is written that way throughout: STGMEDIUM
   holds an unnamed union of handles and reaches into it without naming it. */
#include <stdio.h>

struct outer {
    int tag;
    union { struct { short lo; short hi; }; int both; };
    struct { char a; char b; };
    int tail;
};

union plain { struct { int x; int y; }; long long packed; };

int main(void) {
    struct outer o;
    union plain p;
    o.tag = 9;
    o.both = 0;
    o.lo = 3;
    o.hi = 4;
    o.a = 'x';
    o.b = 'y';
    o.tail = 5;
    p.packed = 0;
    p.x = 11;
    p.y = 22;
    printf("%d %d %d %d %c %c %d %d\n", o.tag, o.lo, o.hi, o.both, o.a, o.b,
           o.tail, (int)sizeof(struct outer));
    printf("%d %d %d\n", p.x, p.y, (int)sizeof(union plain));
    return 0;
}
