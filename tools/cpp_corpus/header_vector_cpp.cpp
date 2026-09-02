/* <vector>: grown, indexed, walked, and shrunk. */
#include <vector>
#include <cstdio>

int main() {
    std::vector<int> held;
    held.push_back(3);
    held.push_back(1);
    held.push_back(2);
    printf("%d %d\n", (int)held.size(), held[1]);
    for (size_t i = 0; i < held.size(); i++) { printf("%d", held[i]); }
    printf("\n");
    held.pop_back();
    printf("%d %d %d\n", (int)held.size(), (int)held.empty(), held.back());
    held.resize(4);
    printf("%d\n", (int)held.size());
    return 0;
}
