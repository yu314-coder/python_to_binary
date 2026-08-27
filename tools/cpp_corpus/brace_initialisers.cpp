#include <cstdio>
struct Point { int x; int y; };
class Counter {
public:
    Counter(int start) : at(start) { }
    int at;
};
int main() {
    wchar_t message[8]{};
    int xs[4]{1, 2, 3, 4};
    char text[6]{'h', 'i', 0};
    Point p{};
    Point q{3, 4};
    int n{};
    int m{7};
    Counter c{5};
    printf("%d %d %d %d %d %d %d %d %s %d\n", (int)message[0], xs[0], xs[3],
           p.x, p.y, q.x, q.y, n, text, m + c.at);
    return 0;
}
