#include <cstdio>
struct Box {
    int slot[3];
    Box() { slot[0] = slot[1] = slot[2] = 0; }
    int &operator[](int i) { return slot[i]; }
    const int &operator[](int i) const { return slot[i]; }
};
static int &pick(Box &b, int i) { return b[i]; }
int main() {
    Box b;
    b[0] = 4;
    pick(b, 1) = 9;
    const Box &r = b;
    printf("%d %d %d\n", b[0], b[1], r[0]);
    return 0;
}
