/* <float.h>: what a double is made of, which is the same everywhere py2bin
   builds because the format is IEEE-754 on all six machines. */
#include <float.h>
#include <stdio.h>

int main(void) {
    printf("%d %d\n", DBL_MANT_DIG, FLT_MANT_DIG);
    printf("%d %d\n", DBL_MAX_EXP, DBL_DIG);
    printf("%d %d\n", DBL_EPSILON > 0.0, DBL_MAX > DBL_MIN);
    printf("%d\n", FLT_RADIX);
    return 0;
}
