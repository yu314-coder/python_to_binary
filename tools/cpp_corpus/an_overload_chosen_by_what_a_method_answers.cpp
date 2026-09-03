// `path` has three constructors taking one argument, and the one meant by
// `dir.c_str()` is the one taking `const char *` - which is what string's
// c_str() is declared to answer, one screen up. The chooser could not read
// it: by the time it asked, the class had been taken apart and its body was
// gone from the text it reads, and it refused with "cast the argument to the
// type of the one you want". The methods had not gone anywhere - each is
// declared under the name the translator gives it - so that is what it reads.
#include <stdio.h>
#include <string>
#include <filesystem>
struct Job {
    std::string dir;
    std::filesystem::path where() { return std::filesystem::path(dir.c_str()); }
};
static std::string tail(const std::string &s) { return s.substr(s.size() - 3); }
int main(void) {
    std::string transferDirectory = "out/dir";
    std::filesystem::path p(transferDirectory.c_str());
    std::filesystem::path q = std::filesystem::path(transferDirectory.c_str()) / "leaf.txt";
    Job j; j.dir = "a/b/c";
    std::filesystem::path chained(tail(j.dir).c_str());
    std::filesystem::path viamethod = j.where();
    printf("%s %s %s %s\n", p.filename().string().c_str(), q.filename().string().c_str(),
           chained.string().c_str(), viamethod.filename().string().c_str());
    return 0;
}
