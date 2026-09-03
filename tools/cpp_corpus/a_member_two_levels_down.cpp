// A method called, and a subscript taken, on a member of a member.
//
// `b.settings.title.size()` is a call on an object two members down from a
// local; py2bin named the members of a local object one level deep, so the
// call reached the C as a method on a struct and was refused. Through an
// object, through a pointer, three levels down, and the same thing from
// inside a method of the outer class, which already worked and must keep
// working: a string's size and comparison, a vector's push_back, size and
// subscript, and a map's subscript.
#include <stdio.h>
#include <string>
#include <vector>
#include <map>

struct Settings {
    std::string title = "Sidecar";
    std::vector<int> ports;
    std::map<std::string, int> counts;
};

struct Bridge {
    Settings settings;
    int width() { return (int)settings.title.size(); }
    int first() { settings.ports.push_back(4); return settings.ports[0]; }
};

struct Host {
    Bridge bridge;
    int spare;
};

int main(void) {
    Bridge b;
    Bridge *p = &b;
    printf("%d %d %d\n", (int)b.settings.title.size(), b.settings.title == "Sidecar", p->settings.title != "Sidecar");
    b.settings.ports.push_back(8080);
    p->settings.ports.push_back(8081);
    b.settings.ports[1] = b.settings.ports[1] + 1;
    printf("%d %d %d\n", b.settings.ports[0], p->settings.ports[1], (int)b.settings.ports.size());
    b.settings.counts["k"] = 3;
    b.settings.counts["k"] += 4;
    printf("%d %d %d\n", b.settings.counts["k"], b.width(), b.first());
    Host h;
    h.bridge.settings.ports.push_back(9);
    h.bridge.settings.title = "Host";
    printf("%d %d %d %s\n", (int)h.bridge.settings.title.size(), h.bridge.settings.ports[0], (int)h.bridge.settings.ports.size(), h.bridge.settings.title.c_str());
    return 0;
}
