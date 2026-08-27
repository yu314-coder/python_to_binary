#include <cstdio>
class Host {
public:
    static int add(int a, int b) { return a + b; }
    int viaPointer(int a, int b) {
        int (*fn)(int, int) = &Host::add;
        int (*bare)(int, int) = Host::add;
        return fn(a, b) + bare(a, b);
    }
};
int main() {
    Host h;
    int (*top)(int, int) = &Host::add;
    printf("%d %d\n", h.viaPointer(3, 4), top(10, 1));
    return 0;
}
