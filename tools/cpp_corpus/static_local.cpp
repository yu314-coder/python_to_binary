#include <stdio.h>
static int counted(){ static int n = 0; n++; return n; }
int main(){ counted(); counted(); printf("%d\n", counted()); return 0; }
