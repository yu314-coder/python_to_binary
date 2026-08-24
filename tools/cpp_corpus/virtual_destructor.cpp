#include <stdio.h>
int freed = 0;
class Base {
public:
    virtual ~Base() { freed = freed + 1; }
    virtual int tag() { return 1; }
};
class Sub : public Base {
public:
    ~Sub() { freed = freed + 10; }
    int tag() { return 2; }
};
int main(void) {
    Base *a = new Sub;
    Base *b = new Base;
    printf("%d %d ", a->tag(), b->tag());
    delete a;
    delete b;
    printf("%d\n", freed);
    return 0;
}
