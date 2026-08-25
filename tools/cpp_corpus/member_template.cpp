#include <stdio.h>
class C { public: template<typename T> T twice(T v){ return v + v; } };
int main(){ C c; printf("%d %.1f\n", c.twice(3), c.twice(1.5)); return 0; }
