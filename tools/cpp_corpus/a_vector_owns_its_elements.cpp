// What a vector does to its elements and in what order: libc++'s, which is
// the vector a program on this machine is compiled against. Growing builds the
// new element first and then moves the old ones; insert with room builds a new
// last element and assigns the rest up; erase assigns down and takes the last
// apart; resize builds each new one from nothing; a copy has storage of its own.
#include <cstdio>
#include <vector>
struct C {
    int v;
    C() : v(0) { printf("default\n"); }
    C(int v) : v(v) { printf("make %d\n", v); }
    C(const C &o) : v(o.v) { printf("copy %d\n", o.v); }
    C &operator=(const C &o) { v = o.v; printf("assign %d\n", o.v); return *this; }
    ~C() { printf("drop %d\n", v); }
};
int main() {
    std::vector<C> v;
    C a(1), b(2), c(3);
    printf("-- push 1\n"); v.push_back(a);
    printf("-- push 2\n"); v.push_back(b);
    printf("-- push 3\n"); v.push_back(c);
    printf("-- cap %zu\n", v.capacity());
    printf("-- insert front\n"); v.insert(v.begin(), a);
    printf("-- erase 1\n"); v.erase(v.begin() + 1);
    printf("-- pop\n"); v.pop_back();
    printf("-- resize 4\n"); v.resize(4);
    printf("-- copy\n"); std::vector<C> w = v;
    printf("-- assign smaller\n"); std::vector<C> x; x.push_back(c); x = w;
    printf("-- cap %zu %zu %zu\n", v.capacity(), w.capacity(), x.capacity());
    printf("-- end\n");
    return 0;
}
