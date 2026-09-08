// `return written > 0 ? made : std::string{};` - two shapes at once, and
// py2bin had neither.
//
// `string{}` is a value of that class, built empty, standing where a value
// goes. C has no such expression - an object needs somewhere to live - so it
// is declared at the top of the statement and named where the braces were.
// Safe to lift out of an arm of a `?:` precisely because there is nothing
// inside the braces to evaluate.
//
// And then the conditional itself. py2bin's C stage lowers one through a slot
// holding a single machine word, so two structs are refused there rather than
// half-copied. What C++ means is nothing harder - one of the two, whichever
// the condition picks - so where the whole answer is a conditional it is
// written as that: an object, an `if` that gives it one answer, an `else`
// that gives it the other. Only there: a conditional inside a larger
// expression would have to be lifted out, and lifting changes when the arms
// are evaluated, which is the one thing about `?:` a program can depend on.
#include <cstdio>
#include <string>
#include <vector>

struct Point {
    int x;
    int y;
    Point() { x = 0; y = 0; }
    Point(int a, int b) { x = a; y = b; }
};

// A class head ends in a colon, a class name and a pair of braces, which is
// exactly the shape above and is not one. Read as an empty object it took
// the base out and left `struct Empty : __py2bin_empty_1;`.
struct Empty : Point {};

static std::string named(int n) {
    std::string made = "abc";
    return n > 0 ? made : std::string{};
}

static Point placed(int n) {
    Point here(3, 4);
    return n > 0 ? here : Point{};
}

static std::vector<int> filled(int n) {
    std::vector<int> made;
    made.push_back(7);
    return n > 0 ? made : std::vector<int>{};
}

// The arms may be objects without any braces in sight.
static std::string longer(const std::string &one, const std::string &two) {
    return one.size() >= two.size() ? one : two;
}

// And a conditional whose arms are numbers is left exactly as it was.
static int bigger(int a, int b) { return a > b ? a : b; }

int main() {
    std::string one = named(1);
    std::string none = named(0);
    Point here = placed(1);
    Point nowhere = placed(0);
    Empty empty;
    printf("%s|%d|%d %d|%d %d|%d %d|%s|%d|%d\n", one.c_str(), (int)none.size(),
           here.x, here.y, nowhere.x, nowhere.y,
           (int)filled(1).size(), (int)filled(0).size(),
           longer("longer", "no").c_str(), bigger(3, 9), empty.x);
    return 0;
}
