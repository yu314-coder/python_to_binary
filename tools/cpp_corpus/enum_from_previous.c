/* How every enum in a generated COM header is written: each entry is the one
   before it plus one, and the "all bits" entry of a flag enum is 0xffffffff -
   past the signed range, and taken by every real compiler. */
#include <stdio.h>

enum Kind { NONE = 0, FIRST = ( NONE + 1 ), SECOND = ( FIRST + 1 ), THIRD };
enum Flags { NO_FLAGS = 0, ONE = 0x1, TWO = 0x2, EVERY = 0xffffffff };

static int table[SECOND + 3];

static int pick(int n) {
    switch (n) {
        case NONE: return 100;
        case FIRST: return 200;
        case SECOND: return 300;
        default: return -1;
    }
}

int main(void) {
    unsigned int mask = (unsigned int)EVERY;
    printf("%d %d %d %d\n", NONE, FIRST, SECOND, THIRD);
    printf("%u %d %u\n", mask, (int)EVERY, (unsigned)(EVERY & TWO));
    printf("%d %d %d %d\n", pick(0), pick(1), pick(2),
           (int)(sizeof table / sizeof table[0]));
    return 0;
}
