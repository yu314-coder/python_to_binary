#include <stdio.h>
int main(void){ unsigned int x = 0xF0;
  printf("%u %u %u %u %u\n", x & 0x30, x | 0x0F, x ^ 0xFF, x >> 4, x << 1); return 0; }
