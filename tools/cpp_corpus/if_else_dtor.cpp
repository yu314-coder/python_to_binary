#include <stdio.h>
class D { public: int v; D() { v = 1; } ~D() { printf("dead\n"); } };
int pick(int n) { if (n > 0) { D d; int r = d.v; return r; } return 0; }
int main(void) { printf("%d\n", pick(1)); printf("%d\n", pick(-1)); return 0; }
