#include <stdio.h>
#pragma pack(push, 1)
struct W { char a; int b; };
#pragma pack(pop)
struct N { char a; int b; };
int main(void) { printf("%d %d\n", (int)sizeof(struct W), (int)sizeof(struct N)); return 0; }
