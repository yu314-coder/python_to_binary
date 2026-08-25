#include <stdio.h>
class W {
public:
    int total; int scale;
    W() { total = 0; scale = 2; }
    void add(int n) { total += n * scale; }
    int go() {
        int extra = 1;
        auto a = [this](int n) { add(n); };
        auto b = [&](int n) { add(n); total += extra; };
        auto c = [=]() { return total + scale; };
        a(3); b(4);
        return c();
    }
};
int main() { W w; printf("%d\n", w.go()); return 0; }
