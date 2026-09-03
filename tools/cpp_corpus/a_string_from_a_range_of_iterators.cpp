// A string built from two iterators, and iterators on a string at all.
//
// `std::wstring(s.begin(), s.end())` is how a program widens a narrow string
// for an interface that wants wide characters, and `std::string(w.begin(),
// w.end())` is how it narrows the answer back. py2bin's own <string> had no
// begin, no end and no range constructor, so `s.begin()` was a member the
// struct did not have. Widening, narrowing, a sub-range of the same width,
// a range from a vector, a range-for and an explicit iterator loop, and the
// two-argument fill constructor still chosen when both arguments are values.
#include <stdio.h>
#include <string>
#include <vector>

static std::wstring widen(const std::string &s) { return std::wstring(s.begin(), s.end()); }
static std::string narrow(const std::wstring &w) { return std::string(w.begin(), w.end()); }

int main(void) {
    std::string s = "title";
    std::wstring w = widen(s);
    std::string back = narrow(w);
    printf("%d %d %s\n", (int)w.size(), (int)back.size(), back.c_str());
    std::string whole = "abcdef";
    std::string head(whole.begin(), whole.begin() + 3);
    std::string tail(whole.begin() + 3, whole.end());
    printf("%s %s %d\n", head.c_str(), tail.c_str(), (int)(whole.end() - whole.begin()));
    int total = 0;
    for (char c : whole) total += c;
    for (std::string::iterator it = whole.begin(); it != whole.end(); ++it) *it = (char)(*it - 32);
    printf("%d %s\n", total, whole.c_str());
    std::vector<char> v;
    v.push_back(104);
    v.push_back(105);
    std::string from(v.begin(), v.end());
    std::wstring filled(5, L'x');
    printf("%s %d\n", from.c_str(), (int)filled.size());
    return 0;
}
