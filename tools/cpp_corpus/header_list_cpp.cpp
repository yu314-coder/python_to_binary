/* <list> as py2bin has it: pushed at both ends and read at both ends. It has
   no `iterator` typedef, so nothing here walks one. */
#include <list>
#include <cstdio>

int main() {
    std::list<int> held;
    held.push_back(2);
    held.push_front(1);
    held.push_back(3);
    printf("%d %d %d\n", (int)held.size(), held.front(), held.back());
    held.pop_front();
    printf("%d %d\n", (int)held.size(), held.front());
    held.pop_back();
    printf("%d %d\n", (int)held.size(), held.back());
    return 0;
}
