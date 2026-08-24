#include <stdio.h>
int main(void){ int n = 3; auto f = [](int x){ return x*2; }; printf("%d\n", f(n)); return 0; }
