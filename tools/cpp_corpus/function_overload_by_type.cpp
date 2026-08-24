#include <stdio.h>
int add(int a) { return a + 1; }
int add(int a, int b) { return a + b; }
double add(double a, double b) { return a + b + 0.5; }
int main(void) { printf("%d %d %.1f\n", add(1), add(2, 3), add(1.0, 2.0)); return 0; }
