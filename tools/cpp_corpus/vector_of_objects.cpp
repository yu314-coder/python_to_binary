#include <stdio.h>
#include <vector>
class Item {
public:
    int id;
    Item() { id = -1; }
    int get() { return id; }
};
int main(void) {
    std::vector<Item> v;
    for (int i = 0; i < 12; i++) { Item made; made.id = i * 5; v.push_back(made); }
    int total = 0;
    for (unsigned long i = 0; i < v.size(); i++) total = total + v[i].get();
    printf("%lu %d %d\n", (unsigned long)v.size(), total, v[11].id);
    return 0;
}
