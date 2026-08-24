#include <stdio.h>
enum class Mode { Off, On };
int main(void){ Mode m = Mode::On; printf("%d\n", (int)m); return 0; }
