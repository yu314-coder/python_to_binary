/* <uchar.h>: the two character types C says are exactly sixteen and
   thirty-two bits wide. macOS has no such header, so clang cannot build this
   one and the sweep compares nothing - it is here for the six targets. */
#include <uchar.h>
#include <stdio.h>

int main(void) {
    char16_t narrow = u'A';
    char32_t wide = U'B';
    printf("%d %d\n", (int)narrow, (int)wide);
    printf("%d %d\n", (int)sizeof(char16_t), (int)sizeof(char32_t));
    return 0;
}
