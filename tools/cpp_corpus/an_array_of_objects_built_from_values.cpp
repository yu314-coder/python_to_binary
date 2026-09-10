// `const std::string values[] = {"sbp=3", "build=windows", made + more};`
// - an array of objects whose elements are not objects. C++ builds each
// one from what is written for it; copied instead, the C read `values[0] =
// *&"sbp=3";`, and the address of a literal is not something to take.
#include <vector>
#include <string>
#include <cstdint>
#include <cstdio>

using Bytes = std::vector<std::uint8_t>;

static void appendDnsName(Bytes &output, const std::string &name) {
    output.push_back(static_cast<std::uint8_t>(name.size()));
    output.insert(output.end(), name.begin(), name.end());
}

static void appendRecord(Bytes &output, const std::string &owner, int type,
                         const Bytes &rdata) {
    appendDnsName(output, owner);
    output.push_back(static_cast<std::uint8_t>(type));
    output.insert(output.end(), rdata.begin(), rdata.end());
}

int main() {
    Bytes response;
    std::string instance = "SidecarBridge";
    Bytes ptr;
    appendDnsName(ptr, instance + "._sb-direct._tcp.local");
    appendRecord(response, "_sb-direct._tcp.local", 12, ptr);
    appendRecord(response, instance + "._sb-direct._tcp.local", 33, ptr);
    std::printf("%d\n", (int)response.size());

    const std::string values[] = {"sbp=3", "build=windows",
                                  std::string("hosts=") + "10.0.0.4"};
    Bytes txt;
    for (const std::string &value : values) {
        txt.push_back(static_cast<std::uint8_t>(value.size()));
        txt.insert(txt.end(), value.begin(), value.end());
    }
    std::printf("%d\n", (int)txt.size());
    return 0;
}
