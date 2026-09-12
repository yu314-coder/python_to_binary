// `value.back()` and `value.pop_back()` - the last character of a string and
// taking it off, which is how a program trims from the end: walking back
// off a UTF-8 continuation byte, or counting the `=` padding of base64. The
// shipped <string> had neither.
#include <string>
#include <cstdio>
static std::string trimmed(std::string value) {
    while (!value.empty() && (static_cast<unsigned char>(value.back()) & 0xC0) == 0x80) {
        value.pop_back();
    }
    return value;
}
int main() {
    std::string encoded = "YWJj==";
    int padding = 0;
    if (!encoded.empty() && encoded.back() == '=') ++padding;
    encoded.pop_back();
    if (!encoded.empty() && encoded.back() == '=') ++padding;
    std::printf("%d %s\n", padding, encoded.c_str());
    std::string cut = "ab\xC3";
    std::printf("%d %d\n", (int)trimmed(cut).size(), (int)trimmed("abc").size());
    std::wstring wide = L"xy";
    std::printf("%d\n", (int)wide.back());
    wide.pop_back();
    std::printf("%d\n", (int)wide.size());
    return 0;
}
