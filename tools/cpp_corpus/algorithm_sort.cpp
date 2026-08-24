#include <stdio.h>
#include <algorithm>
#include <vector>
int main(void) {
    std::vector<int> v;
    v.push_back(5); v.push_back(1); v.push_back(4); v.push_back(2);
    std::sort(v.begin(), v.end());
    int a = 3, b = 9;
    printf("%d %d %d ", std::max(a, b), std::min(a, b), v[0]);
    std::swap(a, b);
    printf("%d %d %lu\n", a, b, (unsigned long)v.size());
    return 0;
}
