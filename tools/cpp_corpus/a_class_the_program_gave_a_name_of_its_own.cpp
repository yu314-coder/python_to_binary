// `using Bytes = std::vector<uint8_t>;` and then `Bytes(first, last)`. The
// alias is a typedef by the time the template copies exist, and the reader
// that types an expression looks a class body up by name - so `hello.begin()`
// on a `Bytes` had no type, the range constructor's copy was never made, and
// the build site was refused for taking two arguments where the class had a
// constructor taking none.
#include <vector>
#include <string>
#include <cstdint>
#include <cstdio>

using Bytes = std::vector<std::uint8_t>;
using Names = std::vector<std::string>;

constexpr char kContext[] = "SidecarBridge-LAN-v3";

static int total(const Bytes &held) {
    int sum = 0;
    for (std::size_t index = 0; index < held.size(); index++) {
        sum += held[index];
    }
    return sum;
}

int main() {
    Bytes hello;
    for (int index = 0; index < 6; index++) {
        hello.push_back((std::uint8_t)(index + 1));
    }
    Bytes tail(hello.begin() + 1, hello.end());
    std::printf("%d %d\n", (int)tail.size(), total(tail));

    Bytes whole(hello.begin(), hello.end());
    std::printf("%d %d\n", (int)whole.size(), total(whole));

    // The same constructor asked for again with the qualifier written in.
    // C++ makes a copy per spelling and C has one type for both, so the two
    // copies were one function defined twice.
    Bytes salt(reinterpret_cast<const std::uint8_t *>(kContext),
               reinterpret_cast<const std::uint8_t *>(kContext)
                   + sizeof(kContext) - 1);
    std::printf("%d %d\n", (int)salt.size(), (int)salt[0]);

    // A string built from the same pair.
    std::string text(hello.begin() + 1, hello.end());
    std::printf("%d\n", (int)text.size());

    // And an alias of another instantiation, so one alias does not answer
    // for the next.
    Names names;
    names.push_back("first");
    names.push_back("second");
    std::printf("%d %s\n", (int)names.size(), names[1].c_str());
    return 0;
}
