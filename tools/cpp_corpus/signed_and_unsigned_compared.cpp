#include <cstdio>
#include <vector>
int main() {
    std::vector<int> v;
    v.push_back(10); v.push_back(20);
    int i = -1;
    unsigned u = 1;
    long big = 3000000000L;
    int found = 0;
    for (size_t k = 0; k < v.size(); ++k) found += v[k];
    printf("%d %d %ld %d %d\n", (int)(i < (int)u), (int)((unsigned)i > u), big,
           found, (int)(v.size() - 1));
    return 0;
}
