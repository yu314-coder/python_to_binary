#include <stdio.h>
class Outer { public: class Inner { public: int v; Inner(){v=9;} }; Inner in; Outer(){} };
int main(void){ Outer o; printf("%d\n", o.in.v); return 0; }
