#include <cstdio>
#include <string>
#include <sstream>
int main() {
    std::istringstream a("7 -8 3.5 -0.25 1e3 hello z");
    int i = 0, j = 0; double d = 0, e = 0, f = 0; std::string w; char c = 0;
    a >> i >> j >> d >> e >> f >> w >> c;
    std::istringstream b("1 2 3 4");
    int total = 0, one = 0;
    while (b >> one) total += one;
    std::istringstream lines("alpha\nbeta\ngamma");
    std::string got; int count = 0; std::string last;
    while (std::getline(lines, got)) { ++count; last = got; }
    printf("%d %d %g %g %g %s %c|%d|%d %s\n", i, j, d, e, f, w.c_str(), c, total, count, last.c_str());
    return 0;
}
