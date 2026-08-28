#include <cstdio>
#include <cstdarg>
static int total(int count, ...) {
    va_list ap; va_start(ap, count);
    int sum = 0;
    for (int i = 0; i < count; i++) sum += va_arg(ap, int);
    va_end(ap);
    return sum;
}
int main() { printf("%d %d\n", total(3, 1, 2, 3), total(5, 10, 20, 30, 40, 50)); return 0; }
