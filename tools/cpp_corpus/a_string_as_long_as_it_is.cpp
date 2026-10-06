// A string as long as it is: a thousand characters, six hundred, both together. It was an array of 256 and every operation stopped at the 255th character without a word.
#include <cstdio>
#include <string>
int main() {
    std::string s;
    for (int i = 0; i < 1000; i++) s += (char)('a' + i % 26);
    std::string t(600, 'x');
    std::string u = s + t;
    std::string w = "0123456789";
    for (int k = 0; k < 6; k++) w = w + w;
    printf("%zu %zu %zu %zu %c %c\n", s.size(), t.size(), u.size(), w.size(), s[999], u[1599]);
    return 0;
}
