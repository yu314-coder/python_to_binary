#include <stdio.h>
int main(void){ double d = 3.9; int n = static_cast<int>(d);
  const int c = 7; int *m = const_cast<int *>(&c); printf("%d %d\n", n, *m); return 0; }
