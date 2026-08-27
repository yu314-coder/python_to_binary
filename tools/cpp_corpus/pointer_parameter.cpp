#include <cstdio>
class Source {
public:
    virtual int value() = 0;
    virtual ~Source() { }
};
class Fixed : public Source {
public:
    int held;
    Fixed() { held = 7; }
    int value() { return held; }
};
class Reader {
public:
    int twice(Source *from) { return from->value() * 2; }
};
int main() {
    Fixed f;
    Reader r;
    printf("%d\n", r.twice(&f));
    return 0;
}
