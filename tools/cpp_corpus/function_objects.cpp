#include <stdio.h>
#include <vector>
#include <algorithm>
#include <functional>
int main(void) {
    std::vector<int> v;
    v.push_back(3); v.push_back(1); v.push_back(2);
    std::greater<int> down;
    std::sort(v.begin(), v.end(), down);
    printf("%d%d%d ", v[0], v[1], v[2]);
    std::plus<int> add;
    std::less<int> up;
    printf("%d %d %d\n", add(2, 3), (int)up(1, 2), (int)down(1, 2));
    return 0;
}
