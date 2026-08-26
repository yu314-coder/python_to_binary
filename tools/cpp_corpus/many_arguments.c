#include <stdio.h>

static int twelve(int a, int b, int c, int d, int e, int f,
                  int g, int h, int i, int j, int k, int l) {
    return a + b * 2 + c * 3 + d * 4 + e * 5 + f * 6
         + g * 7 + h * 8 + i * 9 + j * 10 + k * 11 + l * 12;
}

static double mixed(double a, int b, double c, int d, double e, int f,
                    double g, int h, double i, int j) {
    return a + b + c + d + e + f + g + h + i + j;
}

int main(void) {
    printf("%d\n", twelve(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12));
    printf("%.1f\n", mixed(1.5, 2, 3.5, 4, 5.5, 6, 7.5, 8, 9.5, 10));
    return 0;
}
