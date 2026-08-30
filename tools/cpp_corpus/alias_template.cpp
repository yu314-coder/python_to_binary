#include <cstdio>
#include <vector>
template <typename T> using Row = std::vector<T>;
using Count = int;
int main() { Row<int> r; r.push_back(4); Count c = r[0]; printf("%d\n", c); return 0; }
