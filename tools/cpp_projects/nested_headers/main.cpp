#include <stdio.h>
#include "geometry/point.h"
int main(void) { Point a = { {1, 2} }, b = { {4, 6} }; Vec d = a.to(b); printf("%d %d %d\n", d.x, d.y, d.dot(d)); return 0; }
