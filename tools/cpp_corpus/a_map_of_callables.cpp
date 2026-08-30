#include <cstdio>
#include <map>
#include <string>
#include <functional>
int main() {
    std::map<std::string, std::function<int(int)> > ops;
    ops["double"] = [](int v) { return v * 2; };
    ops["square"] = [](int v) { return v * v; };
    printf("%d %d %d\n", ops["double"](5), ops["square"](5), (int)ops.size());
    return 0;
}
