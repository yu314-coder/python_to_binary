/* <string_view> as py2bin has it: built over a literal, sliced and indexed.
   One built from a `std::string` is not here - the constructor takes a
   `char *`, which is what that subset is. */
#include <string_view>
#include <cstdio>

int main() {
    std::string_view seen("hello world");
    printf("%d\n", (int)seen.size());
    std::string_view part = seen.substr(0, 5);
    printf("%d %c%c\n", (int)part.size(), part[0], seen[6]);
    return 0;
}
