#include <cstdio>
#include <map>
#include <string>
#include <utility>
int main() {
    std::pair<int, std::string> p = std::make_pair(1, std::string("a"));
    std::map<int, std::string> m;
    m.insert(std::make_pair(2, std::string("b")));
    m[3] = "c";
    int keys = 0; int found = 0;
    for (std::map<int, std::string>::iterator it = m.begin(); it != m.end(); ++it) keys += it->first;
    if (m.find(3) != m.end()) found = 1;
    printf("%d %s %d %d %d\n", p.first, p.second.c_str(), keys, found, (int)m.count(9));
    return 0;
}
