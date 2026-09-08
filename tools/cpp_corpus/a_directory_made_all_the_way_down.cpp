// `std::error_code error; std::filesystem::create_directories(p, error);` -
// how a program asks for a directory, does not mind one being there already,
// and does not want a throw either. Four things py2bin did not have:
// `<system_error>`, the plural `create_directories`, the overloads that
// report through an error rather than throwing, and `temp_directory_path`.
//
// And `p /= "piece"`, which is how a path is built up. It is the same member
// `p = p / "piece"` calls written the other way, and the operator table had
// no name for `/=` at all - so it was refused by name, which is at least
// what it should do for something it cannot write.
#include <cstdio>
#include <filesystem>
#include <string>
#include <system_error>

int main() {
    std::filesystem::path root = std::filesystem::temp_directory_path();
    std::filesystem::path made = root;
    made /= "py2bin-corpus-probe";
    made /= std::string("one");
    made /= L"two";

    std::error_code error;
    std::filesystem::create_directories(made, error);
    int there = std::filesystem::is_directory(made) ? 1 : 0;
    int quiet = error ? 0 : 1;

    // Asking again for one that is there is not a failure.
    std::filesystem::create_directories(made, error);
    int again = error.value();

    // Taken away from the bottom up, since a directory holding another
    // cannot go.
    std::filesystem::remove(made, error);
    std::filesystem::path middle = root / "py2bin-corpus-probe" / "one";
    std::filesystem::remove(middle, error);
    std::filesystem::path top = root / "py2bin-corpus-probe";
    std::filesystem::remove(top, error);
    int gone = std::filesystem::exists(top) ? 0 : 1;

    std::error_code plain;
    int empty = plain.value();

    printf("%d %d %d %d %d %d\n", (int)(root.string().size() > 0), there,
           quiet, again, gone, empty);
    return 0;
}
