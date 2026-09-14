/* An include guard holding directives and code together: a header of its own,
   a standard one, a typedef, and a class whose methods name what the headers
   declare. The typedef made the guard count as code, so it stayed below the
   class - and so did the include the class's `time_t` needed. The macro after
   the class is written after code and continued onto a second line: it stays
   where it was written, in one piece. */
#ifndef CLOCK_H
#define CLOCK_H
#include "ticks.h"
#include <string.h>
typedef int reading_t;
class Clock {
public:
    time_t stamp() { return (time_t)42; }
    ticks_t ticks() { return 7; }
    reading_t length(const char *text) { return (reading_t)strlen(text); }
};
#define CLOCK_SCALE \
    3
#endif
