/* `%*d` - the field width handed over as the argument before the value, and
   a negative one meaning the other alignment. */
#include <stdio.h>
int main(void) {
    int w = 6;
    printf("[%*d][%-*d][%*d][%*d]\n", w, 42, w, 42, 3, 42, w, -42);
    printf("[%*s][%-*s][%*c]\n", w, "ab", w, "ab", w, 'z');
    printf("[%*d][%*s][%*c]\n", -5, 7, -4, "ab", -3, 'z');
    printf("[%*d][%*d][%*s]\n", 200, 7, 0, 42, 0, "ab");
    printf("[%*d][%*d]\n", 4, 1, 5, 2);
    return 0;
}
