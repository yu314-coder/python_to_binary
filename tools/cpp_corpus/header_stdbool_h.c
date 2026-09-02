/* <stdbool.h>: three macros, and the one byte the type occupies. */
#include <stdbool.h>
#include <stdio.h>

int main(void) {
    bool yes = true;
    bool no = false;
    printf("%d %d\n", (int)yes, (int)no);
    printf("%d\n", (int)sizeof(bool));
    return 0;
}
