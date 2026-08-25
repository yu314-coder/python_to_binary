#include <stdio.h>
struct F { unsigned int a : 3; unsigned int b : 5; unsigned int c : 24; };
int main(void){ struct F f; f.a = 5; f.b = 20; f.c = 1000;
  printf("%u %u %u %u\n", f.a, f.b, f.c, (unsigned)sizeof(struct F)); return 0; }
