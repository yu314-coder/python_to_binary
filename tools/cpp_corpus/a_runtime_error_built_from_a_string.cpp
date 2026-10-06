// `std::runtime_error("negative: " + std::to_string(v))` keeps the string it was built from, and `what()` says it.
#include <cstdio>
#include <stdexcept>
#include <string>
int parse(int v) { if (v < 0) throw std::runtime_error("negative: " + std::to_string(v)); return v * 2; }
int main() {
    try { printf("%d\n", parse(4)); printf("%d\n", parse(-3)); }
    catch (const std::exception &e) { printf("caught %s\n", e.what()); }
    return 0;
}
