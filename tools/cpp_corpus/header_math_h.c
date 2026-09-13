/* <math.h>: the ones the hardware answers in one instruction, and the ones
   py2bin's own libm evaluates as a polynomial. Printed to four places, which
   is inside what either implementation promises.

   `round()` is here as well, on all six. It breaks ties away from zero, which
   x86-64's roundsd cannot do in one instruction, so there it is worked out
   exactly from trunc() instead. The values below are the ones that tell a
   right answer from the usual shortcut: ties both ways, a negative number
   that rounds to negative zero, and the largest double below one half. */
#include <math.h>
#include <stdio.h>

int main(void) {
    printf("%.4f %.4f\n", sqrt(2.0), fabs(-3.25));
    printf("%.4f %.4f %.4f\n", floor(-1.5), ceil(-1.5), trunc(-1.5));
    printf("%.4f %.4f\n", pow(2.0, 10.0), pow(2.0, 0.5));
    printf("%.4f %.4f\n", exp(1.0), log(10.0));
    printf("%.4f %.4f %.4f\n", sin(1.0), cos(1.0), tan(1.0));
    printf("%.1f %.1f %.1f %.1f %.1f\n", round(2.5), round(-2.5), round(0.5),
           round(-0.3), round(0.49999999999999994));
    return 0;
}
