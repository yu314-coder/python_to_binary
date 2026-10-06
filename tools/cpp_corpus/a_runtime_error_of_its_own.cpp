// A class deriving from `std::runtime_error` passes its message up and adds a member; caught as itself and as the base.
#include <cstdio>
#include <stdexcept>
#include <string>
struct ConfigError : std::runtime_error { int code; ConfigError(const std::string &m, int c) : std::runtime_error(m), code(c) {} };
int main() {
    try { throw ConfigError("missing key", 7); }
    catch (const ConfigError &e) { printf("%s %d\n", e.what(), e.code); }
    try { throw ConfigError("bad value", 9); }
    catch (const std::runtime_error &e) { printf("%s\n", e.what()); }
    return 0;
}
