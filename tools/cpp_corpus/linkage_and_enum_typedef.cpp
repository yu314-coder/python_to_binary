/* `extern "C"` says how a name is to be spelled for a linker. py2bin has one
   translation unit and no linker, so there is nothing for it to say - and its
   braces are not a scope, so what is inside them is at the same level as what
   is outside. A generated COM header wraps the whole of itself in one.

   And `typedef enum M { ... } M;` is C's own way of writing an enum with a
   name; the pass that adds a typedef after an enum body must leave this one
   alone, the declaration not being over at the brace. */
#include <stdio.h>

extern "C" {

typedef
enum Mode
{
    MODE_FIRST = 0,
    MODE_SECOND = ( MODE_FIRST + 1 ),
    MODE_THIRD = ( MODE_SECOND + 1 )
} Mode;

enum Plain { PLAIN_ONE = 1, PLAIN_TWO = 2 };

typedef struct Point { int x; int y; } Point;

extern "C" int doubled(int n);
int doubled(int n) { return n * 2; }

}

int main() {
    Mode m = MODE_THIRD;
    enum Plain p = PLAIN_TWO;
    Point where;
    where.x = 3;
    where.y = 4;
    printf("%d %d %d %d %d\n", (int)m, (int)p, where.x + where.y, doubled(5),
           (int)sizeof(Point));
    return 0;
}
