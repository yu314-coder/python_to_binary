/* `count = count + 1;` in a method of a class that holds `int &count`. A
   reference is never re-seated in C++: an assignment to one writes the
   object it names. It is held as a pointer here, and the pass that puts the
   dereference on every use left the assignments out - so the value was
   stored into the pointer, and the C stage refused it. The only place a
   reference member is bound is the constructor, which is written apart.

   Plain, compound, and a read inside another method, all through the same
   member; and the object it names printed from outside, which is the
   evidence the writes went where C++ says. */
#include <cstdio>

struct Counter {
    int &count;
    explicit Counter(int &c) : count(c) {}
    void bump() { count = count + 1; }
    void set(int v) { count = v; }
    void add(int v) { count += v; }
    int twice() const { return count * 2; }
};

int main() {
    int n = 3;
    Counter c(n);
    c.bump();
    c.set(n * 10);
    c.add(2);
    std::printf("%d %d\n", n, c.twice());
    return 0;
}
