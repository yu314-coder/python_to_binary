// `std::vector<uint8_t> result(n);` - n elements, each value-initialised,
// which is what a program writes when it is about to fill a buffer. The
// shipped <vector> could be built empty or from a range and not from a
// count, and `reserve` takes storage without building anything - so the
// elements would have been whatever was last left where they sit.
#include <vector>
#include <string>
#include <cstdint>
#include <cstdio>

struct Item { int a; int b; };

int main() {
    std::vector<std::uint8_t> result(6);
    std::printf("%d:", (int)result.size());
    for (std::size_t i = 0; i < result.size(); i++) std::printf(" %d", (int)result[i]);
    std::printf("\n");
    result[2] = 9;
    std::printf("%d %d\n", (int)result[2], (int)result[3]);

    std::vector<std::string> names(3);
    names[1] = "second";
    std::printf("%d [%s] [%s]\n", (int)names.size(), names[0].c_str(), names[1].c_str());

    std::vector<Item> items(2);
    std::printf("%d %d %d\n", (int)items.size(), items[0].a, items[1].b);
    return 0;
}
