/* `strtod` and `atof` - reading a number out of text, which py2bin writes
   as C rather than importing: it links nothing it did not compile, and a C
   runtime's strtod lives in a library it has no linker for. What is checked
   here is not only the value but where the reading stopped, which is how a
   caller tells "no number here" from "the number zero". */
#include <stdio.h>
#include <stdlib.h>
int main(void) {
    char *end = 0;
    const char *samples[] = {"3", "-2.5", "  +0.125", "1e3", "2.5e-2", "7e", "abc", "0", "42xyz"};
    int i;
    for (i = 0; i < 9; i++) {
        double v = strtod(samples[i], &end);
        printf("%s -> %.6g, read %d\n", samples[i], v, (int)(end - samples[i]));
    }
    printf("%.6g\n", atof("6.5"));
    return 0;
}
