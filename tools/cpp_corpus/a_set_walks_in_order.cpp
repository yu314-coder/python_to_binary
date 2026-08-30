#include <cstdio>
#include <set>
#include <string>
int main() {
    std::set<int> s;
    s.insert(3); s.insert(1); s.insert(3); s.insert(2);
    int total = 0;
    for (std::set<int>::iterator it = s.begin(); it != s.end(); ++it) total = total * 10 + *it;
    printf("%d %d %d\n", (int)s.size(), total, (int)s.count(3));
    return 0;
}
