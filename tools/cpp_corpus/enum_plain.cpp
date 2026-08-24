#include <stdio.h>
enum Colour { RED, GREEN = 5, BLUE };
int main(void){ Colour c = GREEN; printf("%d %d %d\n", (int)RED, (int)c, (int)BLUE); return 0; }
