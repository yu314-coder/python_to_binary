/* <stdint.h>: the exact-width types, their widths, and the macros that write
   a constant of one. */
#include <stdint.h>
#include <stdio.h>

int main(void) {
    int32_t a = 2000000000;
    uint8_t b = 200;
    int64_t c = INT64_C(9000000000);
    printf("%d %d\n", (int)a, (int)b);
    printf("%d\n", (int)(c / 1000000));
    printf("%d %d %d\n", (int)sizeof(int16_t), (int)sizeof(uint64_t),
           (int)sizeof(intptr_t));
    printf("%d\n", SIZE_MAX > 0);
    return 0;
}
