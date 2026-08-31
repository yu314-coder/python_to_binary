#include <cstdio>
#include <filesystem>
#include <string>
static std::filesystem::path base() { return std::filesystem::path(L"/tmp"); }
static std::filesystem::path join(std::filesystem::path p) { return p / L"web"; }
int main() {
    std::filesystem::path a = join(base());
    printf("%s\n", a.c_str());
    return 0;
}
