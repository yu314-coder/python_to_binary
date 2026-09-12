// Two objects this translator writes for itself, one in a function and one in
// a block inside it. Every counter started at one again for the block, so both
// were called `__py2bin_value_1` - harmless shadowing in C, except that the
// function's destructors are written *inside* the block, at the `return`, and
// there the name means the block's object. The vector was taken apart by a
// destructor holding a string.
#include <string>
#include <vector>
#include <cstdint>
#include <cstdio>

using Bytes = std::vector<std::uint8_t>;

static Bytes made(int n) {
    Bytes b;
    for (int i = 0; i < n; i++) b.push_back((std::uint8_t)(i + 1));
    return b;
}

static std::string label(int n) { return std::string((size_t)n, 'x'); }

static int count(const Bytes &b) { return (int)b.size(); }
static int len(const std::string &s) { return (int)s.size(); }

static int run(int k) {
    int total = count(made(3));
    while (k > 0) {
        total += len(label(2));
        if (total > 100) return -1;
        k--;
    }
    return total;
}

int main() {
    std::printf("%d\n", run(2));
    return 0;
}
