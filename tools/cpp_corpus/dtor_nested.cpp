#include <stdio.h>
class Part { public: int v; Part() { v = 1; } ~Part() { printf("~Part\n"); } };
class Whole { public: Part p; int n; Whole() { n = 2; } ~Whole() { printf("~Whole\n"); } };
int main(void) { Whole w; printf("%d\n", w.n); return 0; }
