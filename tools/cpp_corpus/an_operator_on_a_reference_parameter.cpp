#include <cstdio>
#include <filesystem>
static std::filesystem::path grow(const std::filesystem::path &p) { return p / L"web"; }
int main() { std::filesystem::path a(L"/tmp"); printf("%s\n", grow(a).c_str()); return 0; }
