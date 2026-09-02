/* <inttypes.h>: the format-string pieces, which are what the header is for.
   They are pasted together with the rest of the format at compile time. */
#include <inttypes.h>
#include <stdio.h>

int main(void) {
    int64_t big = 1234567890123LL;
    uint32_t small = 4000000000u;
    printf("%" PRId64 "\n", big);
    printf("%" PRIu32 "\n", small);
    printf("%" PRIx64 "\n", (int64_t)255);
    printf("%" PRIi16 "\n", (int16_t)-7);
    return 0;
}
