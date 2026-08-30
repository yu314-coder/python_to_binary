#include <cstdio>
struct Box {
    int v[3];
    int &at(int i) { return v[i]; }
    const int &at(int i) const { return v[i]; }
    int sum() const { return at(0) + at(1) + at(2); }
};
int main() { Box b; b.at(0) = 1; b.at(1) = 2; b.at(2) = 3; const Box &c = b; printf("%d %d\n", b.sum(), c.at(1)); return 0; }
