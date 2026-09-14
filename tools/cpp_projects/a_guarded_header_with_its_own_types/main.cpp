#include <stdio.h>
#include "clock.h"
int main() {
    Clock c;
    printf("%ld %lu %d %d\n", (long)c.stamp(), (unsigned long)c.ticks(),
           (int)c.length("guarded"), CLOCK_SCALE);
    return 0;
}
