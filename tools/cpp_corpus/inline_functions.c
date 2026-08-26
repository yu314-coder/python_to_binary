#include <stdio.h>

static inline int twice(int n) { return n * 2; }
static __inline int four(int n) { return n * 4; }
static inline const char *label(void) { return "inlined"; }

int main(void) {
    printf("%s %d %d\n", label(), twice(3), four(3));
    return 0;
}
