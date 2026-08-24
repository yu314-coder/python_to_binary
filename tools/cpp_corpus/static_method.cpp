#include <stdio.h>
class M { public: static int twice(int n){ return n*2; } int k; M(){k=1;} };
int main(void){ printf("%d\n", M::twice(21)); return 0; }
