#include <cstdio>
#include <deque>
int main() {
    std::deque<int> d;
    d.push_back(2); d.push_back(3); d.push_front(1); d.push_front(0);
    printf("%d %d %d %d\n", (int)d.size(), d[0], d[3], d.back());
    d.pop_front(); d.pop_back();
    printf("%d %d %d\n", (int)d.size(), d.front(), d.back());
    return 0;
}
