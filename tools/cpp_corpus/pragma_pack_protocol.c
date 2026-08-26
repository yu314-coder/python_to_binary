#include <stdio.h>
#pragma pack(1)
struct H { unsigned char kind; unsigned int length; unsigned short flags; };
#pragma pack()
struct N { unsigned char kind; unsigned int length; unsigned short flags; };
int main(void){ struct H h; h.kind = 1; h.length = 70000; h.flags = 9;
  printf("%d %d %u %u %u\n", (int)sizeof(struct H), (int)sizeof(struct N),
    h.kind, h.length, h.flags); return 0; }
