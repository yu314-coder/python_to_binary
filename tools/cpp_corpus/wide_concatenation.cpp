#include <cstdio>
#include <string>
constexpr wchar_t kHost[] = L"example.local";
static const wchar_t kOther[] = L"other.local";
int main() {
    std::wstring url = std::wstring(L"https://") + kHost + L"/index.html";
    std::wstring two = std::wstring(L"x:") + kOther;
    printf("%d %d\n", (int)url.size(), (int)two.size());
    return 0;
}
