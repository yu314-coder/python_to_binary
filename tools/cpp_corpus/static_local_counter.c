#include <stdio.h>
static int tick(void){ static int n = 10; n++; return n; }
int main(void){ tick(); tick(); printf("%d\n", tick()); return 0; }
