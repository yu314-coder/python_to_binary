#include <stdio.h>
enum class Level : int { Low = 1, High = 9 };
int main(){ printf("%d\n", (int)Level::High); return 0; }
