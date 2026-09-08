// `std::vector<uint8_t> held((std::istreambuf_iterator<char>(file)),
//  std::istreambuf_iterator<char>());` - how a program reads a file whole,
// and py2bin had none of the three pieces. `istreambuf_iterator` is a class
// holding the stream, one character at a time through the stream's own `get`,
// and the empty one is the end: two of them compare equal when both are
// spent. `vector` takes a range, written over `!=` and `++` and nothing else,
// which is all an input iterator promises - so a pair of pointers works here
// too. And a template *constructor* inside a class is expanded now: it is
// never called by name, so the sites are the declarations that build one.
//
// The iterator lives in <fstream>, beside the stream it reads, and not in
// <iterator>: putting it there meant <iterator> had to include <fstream>,
// and `std::size` - which <iterator> declares for anything with a `size()` -
// then had a second reading for every program that asks a container its
// length.
#include <cstdio>
#include <cstdint>
#include <fstream>
#include <filesystem>
#include <iterator>
#include <string>
#include <vector>

int main() {
    const char *where = "a_range_probe.bin";
    {
        std::ofstream made(where, std::ios::binary | std::ios::trunc);
        made.write("abcde", 5);
    }

    std::ifstream file(where, std::ios::binary);
    std::vector<uint8_t> held((std::istreambuf_iterator<char>(file)),
                              std::istreambuf_iterator<char>());

    // The same constructor, given a pair of pointers.
    int numbers[4];
    numbers[0] = 3; numbers[1] = 4; numbers[2] = 5; numbers[3] = 6;
    std::vector<int> copied(&numbers[0], &numbers[0] + 4);

    // And a file with nothing in it, where the two are equal at once.
    const char *empty = "a_range_probe_empty.bin";
    { std::ofstream made(empty, std::ios::binary | std::ios::trunc); }
    std::ifstream nothing(empty, std::ios::binary);
    std::vector<uint8_t> none((std::istreambuf_iterator<char>(nothing)),
                              std::istreambuf_iterator<char>());

    printf("%d %d %d|%d %d %d|%d\n", (int)held.size(), (int)held[0],
           (int)held[4], (int)copied.size(), copied[0], copied[3],
           (int)none.size());

    file.close();
    nothing.close();
    std::filesystem::remove(where);
    std::filesystem::remove(empty);
    return 0;
}
