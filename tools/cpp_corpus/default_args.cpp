#include <stdio.h>
int add(int a, int b = 10) { return a + b; }
int main(void){ printf("%d %d\n", add(1), add(1, 2)); return 0; }
