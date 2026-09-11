// `held.swap(shared);` - two vectors exchanging what they hold, which is
// how a program takes a container's contents away under a lock and then
// does the work outside it. The shipped <vector> had no `swap`.
#include <vector>
#include <string>
#include <cstdio>

static std::vector<int> shared;

static void release() {
    std::vector<int> held;
    held.swap(shared);
    std::printf("held %d shared %d\n", (int)held.size(), (int)shared.size());
    for (std::size_t i = 0; i < held.size(); i++) std::printf(" %d", held[i]);
    std::printf("\n");
}

int main() {
    shared.push_back(3);
    shared.push_back(4);
    shared.push_back(5);
    std::printf("before %d\n", (int)shared.size());
    release();
    std::printf("after %d\n", (int)shared.size());

    std::vector<std::string> one;
    one.push_back("a");
    std::vector<std::string> two;
    two.push_back("b");
    two.push_back("c");
    one.swap(two);
    std::printf("%d %d %s\n", (int)one.size(), (int)two.size(), one[1].c_str());
    return 0;
}
