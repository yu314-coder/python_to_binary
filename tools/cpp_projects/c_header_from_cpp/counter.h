#ifndef COUNTER_H
#define COUNTER_H
typedef struct { int hits; } counter;
void counter_hit(counter *c, int by);
int counter_read(const counter *c);
#endif
