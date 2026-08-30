#include <cstdio>
#include <vector>
#include <functional>
int main() {
    std::vector<std::function<int(int)> > fs;
    fs.push_back([](int v) { return v + 1; });
    fs.push_back([](int v) { return v * 3; });
    printf("%d %d\n", fs[0](4), fs[1](4));
    return 0;
}
