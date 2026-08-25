#include <stdio.h>
typedef union { float f; unsigned int u; } Bits;
int main(void){ Bits b; b.f = 1.0f; printf("%08x\n", b.u); return 0; }
