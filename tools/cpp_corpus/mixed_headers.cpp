#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>
int main(void){ char buf[16]; strcpy(buf, "hi");
  int *p = (int *)malloc(sizeof(int) * 2); p[0] = 3;
  printf("%s %d %.2f %d\n", buf, p[0], sqrt(16.0), abs(-5)); free(p); return 0; }
