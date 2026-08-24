#include <stdio.h>
#include <vector>
#include <algorithm>
int main(void) {
    std::vector<int> v;
    v.push_back(3); v.push_back(1); v.push_back(4); v.push_back(2);
    auto descending = [](int a, int b) { return a > b; };
    std::sort(v.begin(), v.end(), descending);
    for (unsigned long i = 0; i < v.size(); i++) printf("%d", v[i]);
    printf("\n");
    return 0;
}
