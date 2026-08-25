#include <stdio.h>
#include "shape.h"
int main(){ Shape *s = new Square(4); printf("%s %d\n", s->name(), s->area()); delete s; return 0; }
