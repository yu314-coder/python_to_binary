/* <math.h>: the ones the hardware answers in one instruction, and the ones
   py2bin's own libm evaluates as a polynomial. Printed to four places, which
   is inside what either implementation promises.

   `round()` is not here although py2bin has it: it breaks ties away from
   zero and x86-64's roundsd cannot, so it is refused by name on four of the
   six targets and this program is built for all six. */
#include <math.h>
#include <stdio.h>

int main(void) {
    printf("%.4f %.4f\n", sqrt(2.0), fabs(-3.25));
    printf("%.4f %.4f %.4f\n", floor(-1.5), ceil(-1.5), trunc(-1.5));
    printf("%.4f %.4f\n", pow(2.0, 10.0), pow(2.0, 0.5));
    printf("%.4f %.4f\n", exp(1.0), log(10.0));
    printf("%.4f %.4f %.4f\n", sin(1.0), cos(1.0), tan(1.0));
    return 0;
}
