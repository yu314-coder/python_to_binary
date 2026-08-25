#include <stdio.h>
struct F { unsigned int a : 3; unsigned int b : 5; };
int main(){ struct F f; f.a = 5; f.b = 20; printf("%u %u\n", f.a, f.b); return 0; }
