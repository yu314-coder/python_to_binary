#include <cstdio>
#include <list>
int main() {
    std::list<int> l;
    l.push_back(2); l.push_back(3); l.push_front(1);
    printf("%d %d %d\n", (int)l.size(), l.front(), l.back());
    l.pop_front(); l.pop_back();
    printf("%d %d\n", (int)l.size(), l.front());
    return 0;
}
