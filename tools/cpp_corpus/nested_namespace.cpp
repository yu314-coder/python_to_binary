#include <stdio.h>
namespace a::b { int f(){ return 9; } }
int main(){ printf("%d\n", a::b::f()); return 0; }
