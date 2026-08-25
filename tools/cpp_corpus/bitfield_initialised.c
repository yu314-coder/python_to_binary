#include <stdio.h>
struct T { unsigned int a : 5; unsigned int b : 5; };
int main(void){ struct T t = {9, 17}; printf("%u %u\n", t.a, t.b); return 0; }
