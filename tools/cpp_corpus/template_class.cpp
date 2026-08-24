#include <stdio.h>
template<typename T>
class Box {
public:
    T value;
    Box() { value = 0; }
    void set(T v) { value = v; }
    T get() { return value; }
};
template<typename T>
T twice(T v) { return v + v; }
int main(void) {
    Box<int> a; a.set(21);
    Box<double> b; b.set(1.5);
    printf("%d %.1f %d %.1f\n", a.get(), b.get(), twice<int>(5), twice<double>(2.5));
    return 0;
}
