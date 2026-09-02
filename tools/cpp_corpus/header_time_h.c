/* <time.h> as py2bin has it: the types and the tick rate. No clock is read -
   the header declares no functions, which is the whole of what it says. */
#include <time.h>
#include <stdio.h>

int main(void) {
    time_t when = 1000000000;
    clock_t ticks = 500;
    struct tm broken;
    broken.tm_year = 70;
    broken.tm_mon = 0;
    broken.tm_mday = 1;
    printf("%d %d\n", (int)(when / 1000000), (int)ticks);
    printf("%d %d %d\n", broken.tm_year, broken.tm_mon, broken.tm_mday);
    printf("%d\n", (int)(CLOCKS_PER_SEC / 1000));
    printf("%d\n", (int)sizeof(time_t) >= 4);
    return 0;
}
