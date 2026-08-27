#include <cstdio>
#include <filesystem>
#include <string>
static std::filesystem::path base() {
    std::wstring held(6, L'x');
    held.assign(L"/tmp");
    return std::filesystem::path(held);
}
int main() {
    std::filesystem::path a = base() / L"web";
    std::filesystem::path b;
    b = base() / L"other";
    std::filesystem::path c = base();
    c = c / L"third";
    printf("%s %s %s\n", a.c_str(), b.c_str(), c.c_str());
    return 0;
}
