#include <stdio.h>
class Writer {
public:
    virtual int emit(int v) = 0;
    virtual ~Writer() { }
    int emit_twice(int v) { return emit(v) + emit(v); }
};
class Doubler : public Writer {
public:
    int emit(int v) { return v * 2; }
};
class Negator : public Writer {
public:
    int count;
    Negator() { count = 0; }
    int emit(int v) { count = count + 1; return -v; }
};
int main(void) {
    Doubler d; Negator n;
    Writer *w = &d;
    printf("%d ", w->emit_twice(5));
    w = &n;
    printf("%d %d\n", w->emit_twice(5), n.count);
    return 0;
}
