#include <cstdio>
#include <vector>
int main() {
    std::vector<int> v = {1, 2, 3};
    int total = 0;
    for (size_t i = 0; i < v.size(); i++) total += v[i];
    printf("%d %d\n", (int)v.size(), total);
    return 0;
}
