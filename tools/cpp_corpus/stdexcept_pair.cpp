#include <stdio.h>
#include <stdexcept>
#include <utility>
#include <numeric>
#include <vector>
int risky(int n) {
    if (n < 0) throw std::runtime_error("negative");
    return n;
}
int main(void) {
    std::vector<int> v;
    v.push_back(1); v.push_back(2); v.push_back(3);
    printf("%d|", std::accumulate(v.begin(), v.end(), 0));
    std::pair<int, double> p;
    p.first = 4; p.second = 2.5;
    printf("%d %.1f|", p.first, p.second);
    try { risky(-1); }
    catch (std::runtime_error e) { printf("%s|", e.what()); }
    printf("%d\n", risky(7));
    return 0;
}
