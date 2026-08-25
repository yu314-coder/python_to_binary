#include <stdio.h>
struct F { unsigned int a : 3; unsigned int b : 3; };
int main(void){ struct F f; f.a = 1; f.b = 2; f.a += 3; f.b = f.b | 4;
  printf("%u %u\n", f.a, f.b); return 0; }
