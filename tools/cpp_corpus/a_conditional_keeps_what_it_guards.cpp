// The translator moves every preprocessing directive to the top of the file
// it emits, which is right for an include and wrong for a conditional: the
// `#if` and its `#endif` went up while the lines they bracket stayed behind,
// so `#if 0` emitted an empty conditional above the file and three live
// statements in the middle of it, and both arms of an `#ifdef`/`#else` ran.
// The program built and printed what it was never asked to print.
#include <stdio.h>

#ifdef _WIN32
#include <stdlib.h>
#else
#include <string.h>
#endif

#define PICKED 7
#if 0
#define PICKED 99
#endif

struct V {
    int plain() { return 5; }
    int chosen() { return PICKED; }
};

int main(void) {
    V v;
#if 0
    printf("dead one\n");
    printf("dead two\n");
#endif
#ifdef NOT_DEFINED_ANYWHERE
    printf("dead else-arm\n");
#else
    printf("live else-arm\n");
#endif
#if 1
    printf("live if-arm\n");
#else
    printf("dead if-arm\n");
#endif
#if 0
    printf("outer dead\n");
#  if 1
    printf("inner, still dead\n");
#  endif
#endif
#if defined(NOPE)
    printf("dead defined\n");
#elif defined(ALSO_NOPE)
    printf("dead elif\n");
#else
    printf("live elif else\n");
#endif
    printf("plain %d chosen %d\n", v.plain(), v.chosen());
    return 0;
}
