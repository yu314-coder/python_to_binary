/* `for (char c : text) result += c;` in a program that also has a `c` of
   another type, declared above. A range-for declares its variable with a
   `:`, and the reader that types a name did not count that as a
   declaration - so the type it found for `c` was the other one, a `const
   char *`, and `string`'s `+=` was chosen by it. */
#include <cstdio>
#include <string>

static const char *c = "elsewhere";

static std::string copied(const std::string &text) {
    std::string result;
    for (char c : text) {
        result += c;
    }
    return result;
}

int main() {
    std::printf("%s %s\n", copied("Words").c_str(), c);
    return 0;
}
