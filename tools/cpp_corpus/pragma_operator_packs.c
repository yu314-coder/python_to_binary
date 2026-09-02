#include <stdio.h>
#define BEGIN_PACKED _Pragma("pack(push, 1)")
#define END_PACKED   _Pragma("pack(pop)")
#define IGNORED(x)   _Pragma(#x)
IGNORED(acme hint always)
BEGIN_PACKED
struct Wire { unsigned char kind; unsigned int length; };
END_PACKED
struct Loose { unsigned char kind; unsigned int length; };
int main(void){ struct Wire w; w.kind = 3; w.length = 70000;
  _Pragma("acme unroll")
  printf("%d %d %u %u\n", (int)sizeof(struct Wire), (int)sizeof(struct Loose),
    w.kind, w.length); return 0; }
