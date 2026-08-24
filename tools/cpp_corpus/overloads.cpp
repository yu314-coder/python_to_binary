#include <stdio.h>
class N {
public:
    int v;
    N() { v = 0; }
    N(int a) { v = a; }
    N(int a, int b) { v = a + b; }
    int get() { return v; }
};
int main(void) { N x; N y(5); N z(2, 3);
  printf("%d %d %d\n", x.get(), y.get(), z.get()); return 0; }
