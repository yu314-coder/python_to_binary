#include <stdio.h>
template<typename T> int kind(T v){ return 0; }
template<> int kind<double>(double v){ return 1; }
int main(){ printf("%d%d\n", kind(1), kind(1.5)); return 0; }
