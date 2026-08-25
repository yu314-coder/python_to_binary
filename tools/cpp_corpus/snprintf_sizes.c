#include <stdio.h>
int main(void){ char b[4]; int n = snprintf(b, 0, "hello"); int m = snprintf(b, 4, "hello");
  printf("%d %d [%s]\n", n, m, b); return 0; }
