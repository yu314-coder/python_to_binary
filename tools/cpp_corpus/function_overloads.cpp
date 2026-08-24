#include <stdio.h>
int add(int a) { return a + 1; }
int add(int a, int b) { return a + b; }
int add(int a, int b, int c) { return a + b + c; }
int use(int v) { return add(v) + add(v, v) + add(v, v, v); }
int main(void) { printf("%d %d %d %d\n", add(1), add(2, 3), add(1, 2, 3), use(2)); return 0; }
