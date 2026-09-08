// `static const std::map<int, WORD> table = {{0x28, VK_RETURN}, ...};` - how
// a program writes a small lookup table, and py2bin had none of it.
//
// A list fills a container that takes `push_back`; a map and a set take
// `insert`, and what a pair in the list means for a map is the value the key
// stands for. The splitter that reads the entries counted parentheses and
// brackets and not braces, so `{{1, 10}, {2, 20}}` came apart at the comma
// *inside* an entry. And a declaration that begins with `const` is still a
// declaration: read from the `const`, its type was `const`, no class of that
// name takes anything, and the list was left standing.
#include <cstdio>
#include <map>
#include <set>
#include <string>
#include <vector>

typedef unsigned short WORD;

static WORD lookup(int usage) {
    static const std::map<int, WORD> table = {
        {0x28, 13}, {0x29, 27}, {0x2A, 8}
    };
    auto found = table.find(usage);
    return found == table.end() ? 0 : found->second;
}

int main() {
    std::map<int, int> plain = {{1, 10}, {2, 20}};
    const std::map<std::string, int> named = {{"one", 1}, {"two", 2}};
    std::set<int> marks = {3, 1, 2};
    // The shape that has always worked, which must go on working.
    std::vector<int> pushed = {7, 8, 9};

    printf("%d %d %d|%d %d %d|%d %d|%d %d|%d %d %d\n",
           (int)lookup(0x28), (int)lookup(0x2A), (int)lookup(9),
           (int)plain.size(), plain[1], plain[2],
           (int)named.size(), named.at("two"),
           (int)marks.size(), (int)marks.count(2),
           (int)pushed.size(), pushed[0], pushed[2]);
    return 0;
}
