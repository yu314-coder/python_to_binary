#include <cstdio>
class Meter {
public:
    explicit Meter(int start);
    Meter(int a, int b);
    ~Meter();
    int read() const;
    void bump(int by);
private:
    int value_;
};
Meter::Meter(int start) : value_(start) { }
Meter::Meter(int a, int b) : value_(a + b) { }
Meter::~Meter() { }
int Meter::read() const { return value_; }
void Meter::bump(int by) { value_ += by; }
int main() {
    Meter m(4);
    Meter n(2, 3);
    m.bump(6);
    printf("%d %d\n", m.read(), n.read());
    return 0;
}
