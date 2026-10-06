// A wide string past 255 characters, copied, narrowed and widened again.
#include <cstdio>
#include <string>
int main() {
    std::wstring w = L"wide";
    for (int i = 0; i < 300; i++) w += L'x';
    std::wstring copy = w; copy[0] = L'W';
    std::string narrow(w.begin(), w.begin() + 4);
    std::wstring back(narrow.begin(), narrow.end());
    printf("%zu %zu %c %c %s %zu\n", w.size(), copy.size(), (char)w[0], (char)copy[0], narrow.c_str(), back.size());
    return 0;
}
