// `for (auto it = held.rbegin(); it != held.rend(); ++it)` - releasing keys
// in the order opposite to pressing them, which is how a program undoes a
// sequence.
//
// Everywhere else in this subset an iterator is a pointer, and a pointer's
// `++` goes forward - so the one iterator that cannot be a pointer is the one
// that goes the other way. Written as a small class instead, `it !=
// v.rend()` asks a method for an *object*, and a method answering one
// answers nothing in the C: the caller provides the space and the callee
// writes through a hidden pointer, so the call needs a temporary - and a
// temporary is a declaration, which the middle of a `for` header has no room
// for. So the walk is written as what it means, the way a range-`for` is: an
// index counting down, and `*it` in the body is the element at that index.
//
// And `const std::vector<T> &` - how nearly every container is passed - had
// the words in front of the type read as part of its name, so the class
// asked for was `const vector__int`, which nothing declares, and `auto n =
// held.size();` had no type at all.
#include <cstdio>
#include <vector>
#include <string>

static int order[8];
static int at = 0;

static void note(int one) { order[at] = one; at = at + 1; }

static void backwards(const std::vector<int> &held) {
    for (auto it = held.rbegin(); it != held.rend(); ++it) note(*it);
}

static int total(const std::vector<int> &held) {
    auto counted = held.size();
    int sum = 0;
    for (auto it = held.begin(); it != held.end(); ++it) sum += *it;
    return sum + (int)counted;
}

int main() {
    std::vector<int> held;
    held.push_back(1);
    held.push_back(2);
    held.push_back(3);
    backwards(held);

    // `continue` still means the step and then round again.
    int skipped = 0;
    for (auto it = held.rbegin(); it != held.rend(); ++it) {
        int one = *it;
        if (one == 2) { continue; }
        skipped = skipped * 10 + one;
    }

    // And a reverse walk that stops early.
    int stopped = 0;
    for (auto it = held.rbegin(); it != held.rend(); ++it) {
        int one = *it;
        if (one == 2) { break; }
        stopped = one;
    }

    printf("%d %d %d|%d|%d|%d\n", order[0], order[1], order[2],
           total(held), skipped, stopped);
    return 0;
}
