/* A base and members built from values calls answer, in the order C++ builds
   them. What a call writes in front of an initialiser goes with that
   initialiser - including in front of the base, and in front of a member
   whose value reads a member built before it, which has to exist by then.
   The count says how many of the calls ran, and in what order. */
#include <cstdio>
#include <string>

static int calls = 0;

static std::string make(const char *text) {
    ++calls;
    return std::string(text) + std::to_string(calls);
}

static std::string louder(const std::string &earlier) {
    return earlier + "!";
}

struct Base {
    std::string label;
    explicit Base(std::string given) : label(given) {}
    const char *name() const { return label.c_str(); }
};

struct Derived : Base {
    std::string first;
    std::string second;
    int count;
    Derived() : Base(make("b")), first(make("f")), second(louder(first)), count(calls) {}
};

int main() {
    Derived d;
    std::printf("%s %s %s %d\n", d.name(), d.first.c_str(), d.second.c_str(), d.count);
    return 0;
}
