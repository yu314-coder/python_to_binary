#include <cstdio>
#include <cmath>
int main() {
    int a = -7, b = 2;
    double r = std::sqrt(16.0);
    double f = std::floor(-1.5);
    double c = std::fabs(-2.5);
    printf("%d %d %g %g %g %d\n", a / b, a % b, r, f, c, (int)std::pow(2.0, 10.0));
    return 0;
}
