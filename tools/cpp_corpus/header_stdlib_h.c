/* <stdlib.h>: the allocator py2bin ships (an arena, so `free` reclaims
   nothing), the two absolute values and the exit statuses. */
#include <stdlib.h>
#include <stdio.h>

int main(void) {
    int *room = (int *)malloc(4 * sizeof(int));
    int *zeroed = (int *)calloc(4, sizeof(int));
    int i;
    for (i = 0; i < 4; i++) { room[i] = i * i; }
    printf("%d %d %d %d\n", room[0], room[1], room[2], room[3]);
    printf("%d %d\n", zeroed[0], zeroed[3]);
    room = (int *)realloc(room, 8 * sizeof(int));
    printf("%d\n", room[3]);
    free(zeroed);
    printf("%d %ld\n", abs(-17), labs(-17L));
    printf("%d %d\n", EXIT_SUCCESS, EXIT_FAILURE);
    return 0;
}
