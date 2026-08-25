#include <stdio.h>
int main(){ int n = 5; auto f = [v = n * 2]() { return v + 1; }; printf("%d\n", f()); return 0; }
