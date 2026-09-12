// `path directory = length > 0 ? path(buffer) : temp_directory_path();` - a
// conditional as the value of a declaration, which is the third place one
// stands whole. Left for the passes below, they hoist what an arm calls
// into a temporary ahead of the statement and then BOTH arms are
// evaluated - the counter below is what checks that only one is.
#include <filesystem>
#include <string>
#include <cstdio>

static int asked = 0;

static std::filesystem::path fallback() {
    asked += 1;
    return std::filesystem::path("/tmp");
}

int main() {
    int length = 4;
    std::filesystem::path directory = length > 0
        ? std::filesystem::path("/one") : fallback();
    directory /= "SidecarBridge";
    std::printf("%s %d\n", directory.string().c_str(), asked);

    length = 0;
    std::filesystem::path other = length > 0
        ? std::filesystem::path("/one") : fallback();
    std::printf("%s %d\n", other.string().c_str(), asked);
    return 0;
}
