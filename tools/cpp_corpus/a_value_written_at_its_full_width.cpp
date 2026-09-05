// A file offset written to a string stream, and the widths a program holds
// one in. `to_string` went through `int`, so anything past two billion
// printed the bottom half of itself and the program carried on - a file
// longer than two gigabytes reported a negative offset with nothing said.
// The stream now has a form for each width, and `int64_t` reaches the one
// for `long long`: what a fixed-width name stands for is settled by the C
// stage, which is the only place it is settled at all.
#include <cstdio>
#include <cstdint>
#include <string>
#include <sstream>

static std::string described(const std::string &kind, int64_t offset,
                             unsigned long long size) {
    std::ostringstream payload;
    payload << "{\"kind\":\"" << kind << "\",\"offset\":" << offset
            << ",\"size\":" << size << "}";
    return payload.str();
}

int main() {
    printf("%s\n", described("chunk", 4096, 12).c_str());
    printf("%s\n", described("big", 5000000000LL, 18000000000ULL).c_str());
    printf("%s %s\n", std::to_string(-9000000000LL).c_str(),
           std::to_string(12345678901234ULL).c_str());
    return 0;
}
