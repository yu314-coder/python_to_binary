#include <cstdio>
#include <vector>
int main() {
    std::vector<int> v;
    for (int i = 0; i < 6; ++i) v.push_back(i);
    v.erase(v.begin() + 2);
    int total = 0;
    for (size_t i = 0; i < v.size(); ++i) total = total * 10 + v[i];
    v.clear();
    printf("%d %d %d\n", (int)v.size(), total, (int)v.empty());
    return 0;
}
