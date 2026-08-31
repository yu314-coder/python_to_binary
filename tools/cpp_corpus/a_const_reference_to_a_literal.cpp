#include <cstdio>
template <typename T> static T biggest(const T &a, const T &b) { return a < b ? b : a; }
template <typename T> struct Wrap {
    T held;
    Wrap(const T &v) : held(v) {}
    const T &get() const { return held; }
    T copy() const { return held; }
};
int main() {
    Wrap<int> w(7);
    const Wrap<int> &r = w;
    printf("%d %g %d %d\n", biggest(3, 9), biggest(1.5, 0.5), r.get(), w.copy());
    return 0;
}
