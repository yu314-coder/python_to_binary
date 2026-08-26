#include <stdio.h>
class Box { public: int kind; union { int i; float f; } value;
  Box(){ kind = 0; value.i = 0; }
  int get() const { return value.i; } };
int main(){ Box b; b.value.i = 5; printf("%d\n", b.get()); return 0; }
