// Two places an object is given its value as it is declared, and neither was
// read as the construction C++ says it is. At file scope, `std::string dir =
// "x";` matched none of the shapes the file-scope reader knew - `G g;`, `G
// g(7);`, `G table[3];` - so it reached the C as `struct string dir = "x";`,
// which C refuses. On a member, `std::string name = "x";` was written into
// the constructor as `this->name = "x"`, after the body had been rewritten
// and with nothing left below to convert it: a `char *` stored into a struct.
// Both are constructions from the value now, chosen by the value's type.
#include <stdio.h>
#include <string>
#include <filesystem>
namespace fs = std::filesystem;
static std::string transferDirectory = "out/dir";
std::string plainGlobal = "top/level";
struct Job {
    int n = 3;
    std::string name = "job";
    std::string stem = transferDirectory;
    int get() { return n + (int)name.size(); }
};
int main(void) {
    fs::path p(transferDirectory.c_str());
    Job j;
    printf("%s %s %d %s %s\n", p.filename().string().c_str(), plainGlobal.c_str(), j.get(), j.name.c_str(), j.stem.c_str());
    return 0;
}
