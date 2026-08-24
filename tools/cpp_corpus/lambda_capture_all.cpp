#include <stdio.h>
int main(void){
    int a = 2, b = 3, total = 0;
    auto byref = [&](int n) { total += a * n + b; };
    byref(1); byref(2);
    int seen = 0;
    auto byval = [=]() { return a + b + seen; };
    printf("%d %d\n", total, byval());
    return 0;
}
