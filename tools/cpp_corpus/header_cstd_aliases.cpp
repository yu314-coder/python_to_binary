/* The C headers under their C++ names. Which ones exist is not a list py2bin
   keeps: `<cX>` is `<X.h>` asked of whatever C headers it ships, so this
   program says the renaming works rather than that any one name does. */
#include <cassert>
#include <cctype>
#include <cfloat>
#include <cinttypes>
#include <climits>
#include <cmath>
#include <cstdarg>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <cwchar>

static int total(int count, ...) {
    va_list rest;
    int sum = 0;
    va_start(rest, count);
    for (int i = 0; i < count; i++) { sum += va_arg(rest, int); }
    va_end(rest);
    return sum;
}

int main() {
    assert(1);
    printf("%d %d\n", isdigit('4') != 0, CHAR_BIT);
    printf("%.3f %d\n", sqrt(9.0), DBL_DIG);
    printf("%d %d\n", (int)strlen("abcd"), (int)sizeof(size_t));
    printf("%" PRId32 " %d\n", (int32_t)3, total(2, 3, 4));
    int *room = (int *)malloc(sizeof(int));
    room[0] = 5;
    printf("%d\n", room[0]);
    free(room);
    time_t when = 7;
    wchar_t wide[3];
    wide[0] = L'a'; wide[1] = 0;
    printf("%d %d\n", (int)when, (int)wcslen(wide));
    return 0;
}
