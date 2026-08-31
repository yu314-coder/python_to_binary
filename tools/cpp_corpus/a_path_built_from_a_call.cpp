#include <cstdio>
#include <filesystem>
#include <string>
static std::filesystem::path base() {
    std::wstring held(6, L'x');
    held.assign(L"/tmp");
    return std::filesystem::path(held);
}
int main() {
    std::filesystem::path a(base());
    std::filesystem::path b = base() / L"web";
    printf("%s %s\n", a.c_str(), b.c_str());
    return 0;
}
