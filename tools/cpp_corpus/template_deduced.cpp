#include <stdio.h>
template<typename T>
T twice(T v) { return v + v; }
template<typename T>
T bigger(T a, T b) { return a > b ? a : b; }
int main(void) {
    int n = 7;
    double d = 1.25;
    printf("%d %.2f %d %.2f %d\n", twice(5), twice(2.5), bigger(3, 9), bigger(1.5, 0.5), twice(n));
    printf("%.2f\n", twice(d));
    return 0;
}
