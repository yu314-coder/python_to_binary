// `frame.insert(frame.end(), payload.begin(), payload.end());` - a range
// put into a vector at a position, which is how a program joins two of
// them. The shipped <vector> had `insert` for one value and not for a
// range, so a call with a pair of iterators was refused for its count.
#include <vector>
#include <string>
#include <cstdint>
#include <cstdio>

using Bytes = std::vector<std::uint8_t>;

int main() {
    Bytes frame;
    frame.push_back(1);
    frame.push_back(2);
    Bytes payload;
    for (int i = 0; i < 4; i++) payload.push_back((std::uint8_t)(i + 10));

    frame.insert(frame.end(), payload.begin(), payload.end());
    std::printf("%d:", (int)frame.size());
    for (std::size_t i = 0; i < frame.size(); i++) std::printf(" %d", (int)frame[i]);
    std::printf("\n");

    // In the middle, and from part of a range.
    frame.insert(frame.begin() + 1, payload.begin() + 1, payload.begin() + 3);
    std::printf("%d:", (int)frame.size());
    for (std::size_t i = 0; i < frame.size(); i++) std::printf(" %d", (int)frame[i]);
    std::printf("\n");

    // And from a string's bytes.
    std::string text = "abc";
    Bytes held;
    held.insert(held.end(), text.begin(), text.end());
    std::printf("%d %d\n", (int)held.size(), (int)held[2]);
    return 0;
}
