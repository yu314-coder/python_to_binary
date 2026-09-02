/* <stdio.h> as py2bin has it: `printf` and the two that format into a buffer
   are compiled rather than called, and EOF and NULL come out of the header.
   Nothing here opens a file - see the note in the header itself. */
#include <stdio.h>

int main(void) {
    char room[32];
    printf("%s %d %.2f\n", "text", 42, 1.5);
    sprintf(room, "%d-%d", 7, 8);
    printf("%s\n", room);
    snprintf(room, sizeof room, "%c%c%c", 'a', 'b', 'c');
    printf("%s\n", room);
    printf("%d %d\n", EOF, NULL == (void *)0);
    return 0;
}
