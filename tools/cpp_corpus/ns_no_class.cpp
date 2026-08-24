#include <stdio.h>
namespace math {
    int add(int a, int b) { return a + b; }
    int mul(int a, int b) { return a * b; }
}
int main(void) { printf("%d %d\n", math::add(2, 3), math::mul(4, 5)); return 0; }
