#include "shape.h"
Square::Square(int s) { side = s; }
int Square::area() const { return side * side; }
const char *Square::name() const { return "square"; }
