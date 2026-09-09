/* A header's own prototype for a symbol py2bin imports, and a call written
   where the answer is thrown away. Read as a value instead of a statement,
   one whose prototype says `void` was refused for having no value to give -
   which is exactly what the statement had not asked for. */
#include <stdio.h>

#ifdef _WIN32
void Sleep(unsigned int milliseconds);
void SetLastError(unsigned int code);

static void wait_a_moment(void) {
    Sleep(0);
    SetLastError(0);
}
#else
static void wait_a_moment(void) {
}
#endif

int main(void) {
    wait_a_moment();
    printf("%d\n", 7);
    return 0;
}
