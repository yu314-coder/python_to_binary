/* <algorithm>: sorting a vector, the two that pick between a pair, and a
   linear search. */
#include <algorithm>
#include <vector>
#include <cstdio>

int main() {
    std::vector<int> held;
    held.push_back(3);
    held.push_back(1);
    held.push_back(2);
    std::sort(held.begin(), held.end());
    for (size_t i = 0; i < held.size(); i++) { printf("%d", held[i]); }
    printf("\n");
    printf("%d %d\n", std::min(4, 2), std::max(4, 2));
    printf("%d\n", (int)(std::find(held.begin(), held.end(), 2)
                         - held.begin()));
    return 0;
}
