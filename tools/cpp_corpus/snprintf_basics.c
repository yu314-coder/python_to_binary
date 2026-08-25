#include <stdio.h>
#include <string.h>
int main(void){ char b[32]; snprintf(b, sizeof b, "%d-%s-%.2f", 7, "hi", 1.5);
  printf("%s %d\n", b, (int)strlen(b)); return 0; }
