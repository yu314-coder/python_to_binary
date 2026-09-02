/* <numeric> as py2bin has it: `accumulate` over a range. `iota` and `gcd`
   are not declared, which is what this subset is. */
#include <numeric>
#include <vector>
#include <cstdio>

int main() {
    std::vector<int> held;
    held.push_back(1);
    held.push_back(2);
    held.push_back(3);
    printf("%d\n", std::accumulate(held.begin(), held.end(), 0));
    printf("%d\n", std::accumulate(held.begin(), held.end(), 100));
    return 0;
}
