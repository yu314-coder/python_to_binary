#include <stdio.h>
namespace geom {
    class Point {
    public:
        int x; int y;
        Point(int a, int b) { x = a; y = b; }
        int sum() { return x + y; }
    };
    int twice(int n) { return n * 2; }
}
int main(void) {
    geom::Point p(3, 4);
    printf("%d %d\n", p.sum(), geom::twice(5));
    return 0;
}
