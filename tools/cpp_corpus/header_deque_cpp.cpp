/* <deque>: pushed and popped at both ends, and indexed. */
#include <deque>
#include <cstdio>

int main() {
    std::deque<int> held;
    held.push_back(2);
    held.push_front(1);
    held.push_back(3);
    printf("%d %d %d\n", (int)held.size(), held.front(), held.back());
    held.pop_front();
    printf("%d %d\n", held[0], (int)held.size());
    held.pop_back();
    printf("%d\n", held.back());
    return 0;
}
