#include <cstdio>
#include <algorithm>
#include <vector>
struct Item { int id; int weight; };
int main() {
    std::vector<Item> v;
    v.push_back(Item{1, 30});
    v.push_back(Item{2, 10});
    v.push_back(Item{3, 20});
    std::sort(v.begin(), v.end(), [](const Item &a, const Item &b) { return a.weight < b.weight; });
    for (size_t i = 0; i < v.size(); i++) printf("%d:%d ", v[i].id, v[i].weight);
    printf("\n");
    return 0;
}
