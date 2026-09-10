// `f(ok ? "127.0.0.1" : address)` - C++ gives `?:` one type, so the literal
// arm becomes a string and the whole conditional answers one. py2bin lowers
// a conditional through one machine word, so it was handed a `char *` and a
// struct and said so.
//
// Written out as the `if` C++ means. What matters is that the condition is
// asked once and only the arm it picks is evaluated, which the counters
// below check.
#include <string>
#include <cstdio>

static int asked = 0;
static int made = 0;

static bool empty_now(const std::string &value) {
    asked += 1;
    return value.empty();
}

static std::string fallback() {
    made += 1;
    return std::string("127.0.0.1");
}

static std::string joined(const std::string &address, const std::string &name) {
    return address + "/" + name;
}

int main() {
    std::string address;
    std::string instance = "SidecarBridge";

    std::string first = joined(empty_now(address) ? fallback() : address, instance);
    std::printf("%s %d %d\n", first.c_str(), asked, made);

    address = "10.0.0.4";
    std::string second = joined(empty_now(address) ? fallback() : address, instance);
    std::printf("%s %d %d\n", second.c_str(), asked, made);

    // The plain literal arm, which is what their code writes.
    std::string third = joined(address.empty() ? "127.0.0.1" : address, instance);
    std::printf("%s\n", third.c_str());

    // Inside a loop body, where it is lifted once per turn.
    for (int round = 0; round < 3; round++) {
        std::string each = joined(round == 1 ? "one" : address, instance);
        std::printf("%s\n", each.c_str());
    }

    // Both arms the same class.
    std::string left = "L";
    std::string right = "R";
    std::string picked = joined(address.size() > 3 ? left : right, instance);
    std::printf("%s\n", picked.c_str());
    return 0;
}
