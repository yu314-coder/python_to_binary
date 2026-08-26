#include <stdio.h>
#pragma pack(1)
struct P { char a; int b; };
int main(void){ printf("%d\n", (int)sizeof(struct P)); return 0; }
