/* <sys/types.h>: the widths a program takes from the system rather than
   writing out. */
#include <sys/types.h>
#include <stdio.h>

int main(void) {
    size_t width = 12;
    ssize_t signed_width = -3;
    off_t place = 4096;
    printf("%d %d\n", (int)width, (int)signed_width);
    printf("%d\n", (int)place);
    printf("%d\n", (int)sizeof(size_t));
    return 0;
}
