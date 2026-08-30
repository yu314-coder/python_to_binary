#include <cstdio>
#include <map>
#include <string>
int main() {
    std::map<std::string, int> m;
    m["one"] = 1;
    m["two"] = 2;
    m["one"] += 10;
    std::string joined = std::string("a") + "b" + std::string("c");
    std::string n = std::to_string(42);
    printf("%d %d %s %s %d\n", m["one"], (int)m.size(), joined.c_str(), n.c_str(), (int)(joined < "ac"));
    return 0;
}
