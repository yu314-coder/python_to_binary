#include <stdio.h>
int main(){ auto f = [](auto a, auto b){ return a + b; }; printf("%d %d\n", f(1,2), f(10,20)); return 0; }
