#include <stdio.h>
class Log {
public:
    int calls;
    Log() { calls = 0; }
    int show(int v) { calls = calls + 1; return v * 2; }
    int show(double v) { calls = calls + 1; return (int)(v * 10); }
    int show(const char *s) { calls = calls + 1; return s[0]; }
};
int main(void) {
    Log l;
    int n = 4;
    double d = 2.5;
    printf("%d %d %d %d %d\n", l.show(3), l.show(1.5), l.show("A"), l.show(n), l.show(d));
    printf("%d\n", l.calls);
    return 0;
}
