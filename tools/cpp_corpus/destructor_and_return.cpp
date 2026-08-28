#include <cstdio>
static int alive = 0;
class Guard {
public:
    int tag;
    Guard(int t) : tag(t) { alive++; }
    ~Guard() { alive--; }
};
static int scoped() {
    Guard a(1);
    { Guard b(2); if (a.tag) { return alive; } }
    return -1;
}
int main() {
    int inner = scoped();
    printf("%d %d\n", inner, alive);
    return 0;
}
