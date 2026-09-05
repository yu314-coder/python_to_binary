// `cout << s` with s a std::string - the most ordinary line of C++ - and the
// same string beside its size, its characters, and a width to pad to.
#include <iostream>
#include <iomanip>
#include <string>

int main() {
    std::string s = "abc";
    s += "def";
    std::cout << s << " " << s.size() << std::endl;
    std::string t("xy");
    std::cout << "[" << std::setw(5) << t << "]" << t.c_str() << std::endl;
    std::string empty;
    std::cout << "<" << empty << ">" << std::endl;
    return 0;
}
