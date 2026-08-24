#include <stdio.h>
template<typename T>
class Pair {
public:
    T a;
    T b;
    Pair() { a = 0; b = 0; }
    void set(T x, T y) { a = x; b = y; }
    T sum() { return a + b; }
};
template<typename T>
class Holder {
public:
    T item;
    Holder() { }
    void put(T v) { item = v; }
    T take() { return item; }
};
int main(void) {
    Pair<int> p; p.set(3, 4);
    Pair<double> q; q.set(1.5, 2.25);
    Holder<int> h; h.put(99);
    printf("%d %.2f %d\n", p.sum(), q.sum(), h.take());
    return 0;
}
