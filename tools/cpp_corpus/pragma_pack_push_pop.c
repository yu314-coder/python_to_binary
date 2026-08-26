#include <stdio.h>
struct A { char a; int b; };
#pragma pack(push, 1)
struct B { char a; int b; };
#pragma pack(push, 2)
struct C { char a; int b; };
#pragma pack(pop)
struct D { char a; int b; };
#pragma pack(pop)
struct E { char a; int b; };
int main(void){ printf("%d %d %d %d %d\n", (int)sizeof(struct A), (int)sizeof(struct B),
  (int)sizeof(struct C), (int)sizeof(struct D), (int)sizeof(struct E)); return 0; }
