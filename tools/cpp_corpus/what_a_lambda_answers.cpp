// `const Bytes nonce = [&] { Bytes value; randomBytes(value, 32); return value; }();`
// - what a lambda answers is read from its `return`, and the reader was
// asked of the lambda's body and the whole file joined. It takes the last
// declaration of that name anywhere, and `value` is a name a program uses
// in a dozen functions: this one was given the type of a `const wchar_t *`
// written somewhere else entirely.
#include <vector>
#include <string>
#include <cstdint>
#include <cstdio>

using Bytes = std::vector<std::uint8_t>;

static void randomBytes(Bytes &output, std::size_t count) {
    for (std::size_t i = 0; i < count; i++) output.push_back((std::uint8_t)(i + 1));
}

static Bytes make() {
    const Bytes nonce = [&] { Bytes value; randomBytes(value, 4); return value; }();
    return nonce;
}

// A later function giving that name to something else entirely.
static void describe() {
    const wchar_t *value = L"other";
    std::printf("%d\n", (int)value[0]);
}

int main() {
    Bytes n = make();
    std::printf("%d %d\n", (int)n.size(), (int)n[3]);
    describe();
    return 0;
}
