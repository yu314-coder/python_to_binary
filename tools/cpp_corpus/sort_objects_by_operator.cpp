#include <stdio.h>
#include <vector>
#include <algorithm>
class Item { public: int id; Item(int v):id(v){} bool operator<(const Item &o) const { return id < o.id; } };
int main(){ std::vector<Item> v; v.push_back(Item(3)); v.push_back(Item(1));
  std::sort(v.begin(), v.end()); printf("%d%d\n", v[0].id, v[1].id); return 0; }
