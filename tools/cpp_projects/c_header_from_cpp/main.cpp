#include <stdio.h>
extern "C" {
#include "counter.h"
}
struct Meter { counter c; Meter() { c.hits = 0; } void tick() { counter_hit(&c, 3); } };
int main(void) { Meter m; m.tick(); m.tick(); printf("%d\n", counter_read(&m.c)); return 0; }
