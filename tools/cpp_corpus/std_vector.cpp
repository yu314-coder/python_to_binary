#include <stdio.h>
#include <vector>
int main(void) {
    std::vector<int> v;
    for (int i = 0; i < 30; i++) v.push_back(i * i);
    v[3] = 1000;
    std::vector<double> d;
    d.push_back(1.5); d.push_back(2.5);
    unsigned long total = 0;
    for (unsigned long i = 0; i < v.size(); i++) total = total + v[i];
    printf("%lu %lu %d %d %.1f %d\n",
           (unsigned long)v.size(), total, v[3], v.back(), d[1], (int)v.empty());
    return 0;
}
