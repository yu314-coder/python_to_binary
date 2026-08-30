#include <cstdio>
struct Holder {
    int &slot;
    Holder(int &s) : slot(s) {}
    void bump() { ++slot; }
    int read() const { return slot; }
};
int main() {
    int n = 5;
    Holder h(n);
    h.bump(); h.bump();
    printf("%d %d\n", n, h.read());
    return 0;
}
