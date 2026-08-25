#include <stdio.h>
int main(){ int a[5] = {1,2,3,4,5}; int s = 0; for (int x : a) s += x; printf("%d\n", s); return 0; }
