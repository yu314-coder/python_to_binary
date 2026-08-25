#include <stdio.h>
struct S { signed int a : 4; signed int b : 4; };
int main(void){ struct S s; s.a = -3; s.b = 7;
  printf("%d %d %d\n", s.a, s.b, (int)sizeof(struct S)); return 0; }
