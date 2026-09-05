// C++11's brace initialiser on a data member, in every shape a program
// writes one: an atomic, a plain int and double, a string, an aggregate, an
// empty container, a container given a list, a pointer, a class with a
// constructor taking two, and two arrays.
//
// py2bin read the class body before it read the braces: a brace before the
// semicolon opened a method body, so `std::atomic<bool> running_{false};`
// stopped the build with "cannot read the member" - and where an earlier
// pass had already turned the braces into a local's pushes, the member left
// the struct with nothing said. Each brace is now the member's own, and what
// it means is decided by the member's type where the constructor is written.
#include <cstdio>
#include <string>
#include <vector>
#include <atomic>

struct Point { int x; int y; };

class Range {
public:
    int lo; int hi;
    Range(int a, int b) { lo = a; hi = b; }
    int span() { return hi - lo; }
};

class Bridge {
public:
    std::atomic<bool> running_{false};
    std::atomic<int> count_{0};
    int n_{7};
    double ratio_{2.5};
    std::string name_{"sidecar"};
    Point p_{1, 2};
    std::vector<int> v_{};
    std::vector<int> filled_{3, 4, 5};
    Bridge *next_{nullptr};
    Range range_{3, 10};
    char tag_[8]{'a', 'b', 0};
    int grid_[3]{4, 5, 6};

    void start() { running_.store(true); count_.fetch_add(2); }
    int total() { return n_ + p_.x + p_.y + range_.span(); }
};

int main() {
    Bridge b;
    printf("%d %d %s %.1f\n", (int)b.running_.load(), b.n_, b.name_.c_str(), b.ratio_);
    b.start();
    printf("%d %d %d\n", (int)b.running_.load(), b.count_.load(), b.total());
    printf("%d %d %d %d\n", (int)b.v_.size(), (int)b.filled_.size(), b.filled_[2], b.next_ == 0);
    printf("%s %d %d %d\n", b.tag_, b.grid_[0], b.grid_[2], b.p_.y);
    return 0;
}
