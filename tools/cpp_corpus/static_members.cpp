// A static member is a free function that happens to be written in a class:
// it takes no object, so a value return from one needs the hidden pointer
// with nothing in front of it, and a bare call to one must not pass `this`.
#include <cstdio>
#include <string>

struct Holder {
    int value;
};

static Holder makeHolder(const char *label) {
    Holder held;
    held.value = (int)label[0];
    return held;
}

class Make {
public:
    static std::string tag(const char *key) {
        std::string out("<");
        out += key;
        out += ">";
        return out;
    }
    static std::string plain() { return std::string("bare"); }
    static int width(const std::string &of) { return (int)of.size(); }
    std::string wrapped(const char *k) { return tag(k); }
    int twice(const char *k) { return width(tag(k)) * 2; }
};

int main() {
    Make m;
    // A free function answering an object, called with a literal argument:
    // the declaration and the call are one match spanning that literal.
    Holder held = makeHolder("Q");
    printf("%s %s %s %d %d %d\n", Make::tag("a").c_str(), Make::plain().c_str(),
           m.wrapped("z").c_str(), Make::width(Make::tag("bb")), m.twice("c"),
           held.value);
    return 0;
}
