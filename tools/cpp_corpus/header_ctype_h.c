/* <ctype.h>: the classifiers and the two that change case. */
#include <ctype.h>
#include <stdio.h>

int main(void) {
    printf("%d%d%d%d\n", isalpha('q') != 0, isdigit('7') != 0,
           isspace(' ') != 0, isxdigit('f') != 0);
    printf("%c%c\n", (char)toupper('a'), (char)tolower('Z'));
    printf("%d%d%d\n", isupper('A') != 0, ispunct(',') != 0,
           isblank('\t') != 0);
    return 0;
}
