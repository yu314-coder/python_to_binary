/* <set>: one of each, in order. */
#include <set>
#include <cstdio>

int main() {
    std::set<int> held;
    held.insert(3);
    held.insert(1);
    held.insert(3);
    held.insert(2);
    printf("%d\n", (int)held.size());
    for (std::set<int>::iterator it = held.begin(); it != held.end(); ++it) {
        printf("%d", *it);
    }
    printf("\n");
    printf("%d %d\n", (int)held.count(1), (int)held.count(9));
    return 0;
}
