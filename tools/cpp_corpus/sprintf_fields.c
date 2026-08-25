#include <stdio.h>
int main(void){ char b[64]; sprintf(b, "%5.2f|%-6s|%03d", 1.5, "ab", 7);
  printf("[%s]\n", b); return 0; }
