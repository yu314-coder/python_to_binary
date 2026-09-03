/* A C implementation compiled as C++, which is how a C library lands in a
   C++ project more often than not. The header keeps C linkage either side. */
extern "C" {
#include "counter.h"
void counter_hit(counter *c, int by) { c->hits += by; }
int counter_read(const counter *c) { return c->hits; }
}
