// A member template's copies are made from the calls, and which class a
// call is on is read from the receiver. Asked of the whole file rather
// than at the call, the reader takes the last declaration of that name
// anywhere - and `output` is a name two functions apart may both use. The
// `ostringstream output` below answered for the `Bytes &output` here, so
// the call was taken for one on another class and no copy was written.
#include <vector>
#include <string>
#include <sstream>
#include <cstdint>
#include <cstddef>
#include <cstdio>

using Bytes = std::vector<std::uint8_t>;

static void appendDnsName(Bytes &output, const std::string &name) {
    std::size_t start = 0;
    while (start < name.size()) {
        const std::size_t dot = name.find('.', start);
        const std::size_t length =
            dot == std::string::npos ? name.size() - start : dot - start;
        output.push_back(static_cast<std::uint8_t>(length));
        output.insert(output.end(),
                      name.begin() + static_cast<std::ptrdiff_t>(start),
                      name.begin() + static_cast<std::ptrdiff_t>(start + length));
        if (dot == std::string::npos) break;
        start = dot + 1;
    }
    output.push_back(0);
}

// A later function that gives the same name to something else entirely.
static std::string describe(int state) {
    std::ostringstream output;
    output << "state=" << state;
    return output.str();
}

int main() {
    Bytes out;
    appendDnsName(out, "ab.cde");
    std::printf("%d:", (int)out.size());
    for (std::size_t i = 0; i < out.size(); i++) std::printf(" %d", (int)out[i]);
    std::printf("\n%s\n", describe(3).c_str());
    return 0;
}
