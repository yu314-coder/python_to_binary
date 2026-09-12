// A declaration written with braces - `array<uint8_t, 16> tag{};` - and the
// same name used earlier in the unit as a parameter of some other type. The
// reader passed over the brace form, so the type it found for `tag` was the
// other function's, and the range `insert` chosen for `tag.begin()` was the
// one taking characters.
#include <array>
#include <vector>
#include <string>
#include <cstdint>
#include <cstdio>

using Bytes = std::vector<std::uint8_t>;

static void report(const std::string &tag) { std::printf("%s\n", tag.c_str()); }

static Bytes framed(const Bytes &header, const Bytes &ciphertext) {
    std::array<std::uint8_t, 16> tag{};
    for (int i = 0; i < 16; i++) tag[i] = (std::uint8_t)(i + 3);
    Bytes result = header;
    result.insert(result.end(), ciphertext.begin(), ciphertext.end());
    result.insert(result.end(), tag.begin(), tag.end());
    return result;
}

int main() {
    report("note");
    Bytes header; header.push_back(1);
    Bytes ciphertext; ciphertext.push_back(2);
    Bytes out = framed(header, ciphertext);
    std::printf("%d %d %d\n", (int)out.size(), (int)out[1], (int)out[17]);
    return 0;
}
