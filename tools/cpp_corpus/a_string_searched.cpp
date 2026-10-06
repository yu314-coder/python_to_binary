// find, rfind and the find_first_of family, each answering npos - the largest size_t - where there is nothing to find, as C++ says.
#include <cstdio>
#include <string>
int main() {
    std::string s = "the quick brown fox jumps over the lazy dog";
    printf("%zu %zu %zu %zu\n", s.find("the"), s.find("the", 1), s.rfind("the"), s.find("cat"));
    printf("%zu %zu %zu %zu\n", s.find('q'), s.find_first_of("aeiou"), s.find_last_of("aeiou"), s.find_first_not_of("the "));
    printf("%zu %d %d\n", std::string::npos, (int)(s.find("zzz") == std::string::npos), (int)(s.find('x') != std::string::npos));
    printf("%zu %zu\n", s.find_last_not_of("dog"), s.rfind('o', 20));
    return 0;
}
