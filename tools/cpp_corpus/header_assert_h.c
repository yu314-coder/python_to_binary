/* <assert.h>: the macro, and the second form everyone writes where the
   message is an && with a string. */
#include <assert.h>
#include <stdio.h>

int main(void) {
    int n = 3;
    assert(n == 3);
    assert(n > 0 && "n must be positive");
    printf("assert passed n=%d\n", n);
    return 0;
}
