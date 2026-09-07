// `std::array<T, N>` - N elements and nothing else. py2bin shipped it as a
// pointer and a count, on the belief that the size could not be a template
// argument; it can, and the shape mattered. `std::array<uint8_t, 1500>
// buffer{};` is how a program asks for a receive buffer on the stack, and
// against the old one it read and wrote through a null pointer. There is no
// constructor now, so the braces are an aggregate's and mean what C says:
// `{}` zeroes, `{a, b}` fills the first two. The braces on a type still
// spelled with its arguments are read only once the copy of the template has
// been written out, which is later than a declaration is usually read.
#include <cstdio>
#include <array>
#include <cstdint>
#include <string>
#include <vector>

static int sum(const std::array<uint8_t, 4> &given) {
    int total = 0;
    for (unsigned long i = 0; i < given.size(); i++) total += given[i];
    return total;
}

struct Packet {
    std::array<uint8_t, 2> id;
    int length;
};

int main() {
    std::array<uint8_t, 4> header{};
    header[0] = 1;
    header[1] = 2;
    std::array<int, 3> three{7, 8, 9};
    std::array<uint8_t, 8> filled;
    filled.fill(3);
    std::array<uint8_t, 2> pair{header[0], header[1]};

    int walked = 0;
    for (int one : three) walked += one;

    int through = 0;
    for (int *at = three.begin(); at != three.end(); at++) through += *at;

    // The braces the program leaves out: a struct holding an array takes the
    // values in order, which is C's rule and not a special case for arrays.
    Packet packet = {9, 4, 12};

    // Still a container with a list, not an aggregate: the pushes have to
    // win over the braces here.
    std::vector<int> pushed{1, 2, 3};
    std::string named{"hi"};

    printf("%d %d %d %d %d %d %d %d %d %d %d %d %s %d\n",
           (int)header.size(), sum(header), walked, through,
           three.front(), three.back(), (int)filled[7], (int)filled[0],
           (int)pair[0], (int)pair[1],
           (int)packet.id[1], packet.length,
           named.c_str(), (int)sizeof(std::array<uint8_t, 16>));
    printf("%d %d %d\n", (int)pushed.size(), pushed[0], pushed[2]);
    return 0;
}
