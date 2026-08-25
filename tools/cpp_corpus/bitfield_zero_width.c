#include <stdio.h>
struct W { unsigned char a : 2; unsigned char : 0; unsigned char b : 2; };
int main(void){ struct W w; w.a = 3; w.b = 2;
  printf("%u %u %u\n", w.a, w.b, (unsigned)sizeof(struct W)); return 0; }
