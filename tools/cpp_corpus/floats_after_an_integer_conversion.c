/* Everything after the first floating conversion of a printf came out 0.
   The formatter is written out once and jumped into from every conversion
   after it, but the slots inside it were ordinary temporaries - reclaimed at
   the end of whichever statement first formatted a float, and then handed to
   a later statement for its own arguments. Jumping into the formatter wrote
   over them, so `printf("%.3f %.3f", x, y)` printed y as 0.000.

   It needed an earlier conversion to shift the offsets far enough for the
   collision to land on an argument rather than on nothing, which is why no
   corpus program had ever shown it: none put an integer conversion in front
   of a pair of floating ones. Eleven of the thirteen lines below were wrong. */
#include <stdio.h>
int main(void) {
    double a = 4, x = 1.75, y = 1.25, z = -0.5, w = 12345.6789;
    printf("%d\n", 1);
    printf("%.1f\n", a);
    printf("%.3f %.3f\n", x, y);
    printf("%.3f %.3f %.3f %.3f\n", x, y, z, w);
    printf("%d %.2f %d %.2f\n", 7, x, 8, y);
    printf("%8.3f|%-8.3f|%08.3f\n", x, y, z);
    printf("%e %e\n", x, w);
    printf("%g %g %g\n", x, w, z);
    printf("%.0f %.0f %.0f\n", a, x, y);
    printf("%s %.2f %s %.2f\n", "p", x, "q", y);
    printf("%.2f %d %.2f %d %.2f\n", x, 1, y, 2, z);
    printf("%+.2f % .2f %+.2f\n", x, y, z);
    printf("%.17g %.17g\n", x, y);
    return 0;
}
