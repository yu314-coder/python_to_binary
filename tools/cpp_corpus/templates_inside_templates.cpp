#include <cstdio>
#include <vector>
template <typename T> struct Box { T v; Box(T x) : v(x) {} T get() { return v; } };
template <typename T> static int count(const std::vector<T> &v) { return (int)v.size(); }
int main() {
    Box<Box<int> > nested(Box<int>(5));
    std::vector<int> a; a.push_back(1); a.push_back(2);
    std::vector<Box<int> > b; b.push_back(Box<int>(9));
    printf("%d %d %d %d\n", nested.get().get(), count(a), count(b), b[0].get());
    return 0;
}
