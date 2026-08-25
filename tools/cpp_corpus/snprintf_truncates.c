#include <stdio.h>
int main(void){ char b[8]; int n = snprintf(b, sizeof b, "%s-%d", "abcdef", 12345);
  printf("[%s] %d\n", b, n); return 0; }
