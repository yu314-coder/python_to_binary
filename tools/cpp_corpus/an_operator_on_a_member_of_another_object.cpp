// An operator applied to a member of another object.
//
// `s.title += "!"` is `operator+=` on a string that is a member of `s`; the
// operator pass matched a bare name on the left and nothing else, so this
// reached the C as `+=` on a struct and was refused - at one level down and
// at two. Compound assignment, a binary operator whose answer is an object,
// the comparisons, through an object and through a pointer, a member of a
// member, and an operator's answer used as a receiver.
#include <stdio.h>
#include <string>
#include <vector>

struct Settings {
    std::string title = "Sidecar";
    std::vector<int> ports;
};

struct Bridge {
    Settings settings;
    std::string name = "bridge";
};

int main(void) {
    Bridge b;
    Bridge *p = &b;
    b.name += "!";
    p->name += "?";
    std::string more = b.name + "#";
    std::string wide = p->settings.title + more;
    printf("%s %s %s\n", b.name.c_str(), more.c_str(), wide.c_str());
    printf("%d %d %d %d\n", b.name == "bridge!?", p->name != "bridge", b.name < more, b.settings.title == "Sidecar");
    b.settings.title += " app";
    p->settings.title += "s";
    std::string joined = b.settings.title + "/" + b.name;
    printf("%s %s %d\n", b.settings.title.c_str(), joined.c_str(), (int)(b.settings.title + "##").size());
    b.settings.ports.push_back(1);
    b.settings.ports[0] += 41;
    printf("%d %s\n", b.settings.ports[0], (b.name + "@").c_str());
    return 0;
}
