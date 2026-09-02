/* <stddef.h>: the two types, the null pointer and the member offset. */
#include <stddef.h>
#include <stdio.h>

struct Pair { int first; double second; };

int main(void) {
    size_t width = sizeof(struct Pair);
    ptrdiff_t gap;
    int room[4];
    gap = &room[3] - &room[0];
    printf("%d %d\n", (int)width, (int)gap);
    printf("%d\n", (int)offsetof(struct Pair, second));
    printf("%d\n", NULL == (void *)0);
    return 0;
}
