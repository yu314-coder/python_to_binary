// A wide string indexed, compared, searched and cut - as a narrow one is.
//
// py2bin's `wstring` had a constructor, `size`, `c_str`, `assign`, `+` and
// `push_back`, and nothing else its `string` has: `w[i]` on a `const
// std::wstring &` reached the C as a subscript on a struct, so a program
// narrowing a wide string one character at a time was refused with "cannot
// cast a value of type struct wstring to char". Indexed through a reference
// and a local, read and written, compared six ways against a wstring and a
// literal, searched forwards and back for a character and a piece, cut with
// substr, appended and compared.
#include <stdio.h>
#include <string>

static std::string narrow(const std::wstring &w) {
    std::string out;
    for (size_t i = 0; i < w.size(); ++i) out.push_back((char)w[i]);
    return out;
}

int main(void) {
    std::wstring w = L"sidecar";
    std::wstring same = L"sidecar";
    std::wstring other = L"tidecar";
    w[0] = L'S';
    printf("%s %d %d\n", narrow(w).c_str(), (int)w[1], (int)w.size());
    printf("%d %d %d %d %d %d\n", w == same, w != same, w < other, w <= other, w > other, w >= other);
    printf("%d %d\n", same == L"sidecar", same != L"sidecar");
    printf("%d %d %d %d\n", (int)same.find(L'e'), (int)same.find(L"car"), (int)same.find(L'z'), (int)same.rfind(L'a'));
    printf("%d %d\n", (int)same.find(L'a', 6), (int)same.rfind(L"de"));
    printf("%s %s\n", narrow(same.substr(4)).c_str(), narrow(same.substr(1, 3)).c_str());
    std::wstring joined = same;
    joined.append(other);
    joined += L'!';
    printf("%s %d %d\n", narrow(joined).c_str(), same.compare(L"sidecar"), same.compare(L"a") > 0);
    return 0;
}
