#include <cstdio>
#include <vector>
#include <string>
struct P { int x, y; };
int main() {
    std::vector<int> v = { 1, 2, 3, 4 };
    int total = 0;
    for (size_t i = 0; i < v.size(); ++i) total += v[i];
    P p = { 7, 8 };
    int a[3] = { 5, 6, 7 };
    std::vector<std::string> names;
    names.push_back("x"); names.push_back("y");
    printf("%d %d %d %d %s\n", (int)v.size(), total, p.x + p.y, a[2], names[1].c_str());
    return 0;
}
