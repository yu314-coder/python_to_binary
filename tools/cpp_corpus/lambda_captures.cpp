#include <stdio.h>
#include <vector>
#include <algorithm>
int main(void) {
    int base = 10;
    auto add = [base](int x) { return x + base; };
    auto square = [](int x) -> int { return x * x; };
    auto shout = []() { printf("hi "); };
    shout();
    printf("%d %d ", add(5), square(4));
    std::vector<int> v;
    v.push_back(3); v.push_back(1); v.push_back(2);
    std::sort(v.begin(), v.end());
    printf("%d%d%d\n", v[0], v[1], v[2]);
    return 0;
}
