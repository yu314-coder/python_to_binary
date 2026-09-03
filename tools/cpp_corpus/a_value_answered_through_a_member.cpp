// A method that answers an object, called on a member of another object.
//
// `b.s.name()` from outside, and `settings.root()` from inside a method of
// the object holding `settings`: in both the receiver is a member reached
// through something else, and the call was written without the space its
// answer is returned in - the C compiler counted the arguments and refused.
// Also a bare member handed to a constructor with more than one one-argument
// form, `path(base)` inside a method: the type of `this->base` is the type
// the enclosing class declared, not the first `this` in the file.
#include <stdio.h>
#include <string>
#include <filesystem>
namespace fs = std::filesystem;

struct Settings {
    std::string base = "out";
    std::string name() { return std::string("n"); }
    fs::path root() { return fs::path(base); }
    fs::path rootFromC() const { return fs::path(base.c_str()); }
};

struct Bridge {
    Settings settings;
    std::string tag() { return settings.name() + "!"; }
    fs::path target(const std::string &leaf) { return settings.root() / leaf.c_str(); }
    fs::path other(const std::string &leaf) { return this->settings.rootFromC() / leaf; }
};

int main(void) {
    Bridge b;
    Bridge *p = &b;
    std::string t = b.settings.name() + "?";
    fs::path direct = p->settings.root() / "direct.txt";
    printf("%s %s %s\n", b.tag().c_str(), t.c_str(), direct.string().c_str());
    printf("%s %s\n", b.target("leaf.txt").string().c_str(), p->other("o.txt").string().c_str());
    return 0;
}
