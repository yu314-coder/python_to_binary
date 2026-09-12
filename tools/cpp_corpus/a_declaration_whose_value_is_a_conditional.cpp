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

// And one whose two arms do not agree with each other: the declared type is
// what the object is, and an arm the reader cannot work out is left to the
// assignment, which is the pass that knows how to convert one.
static std::string stable(const std::string &deviceID, const std::string &deviceKind,
                          const std::string &deviceName) {
    const std::string stableID = deviceID.empty()
        ? deviceKind + ":" + deviceName
        : deviceID;
    return stableID;
}

int main() {
    std::printf("%s %s\n", stable("", "phone", "mine").c_str(),
                stable("kept", "phone", "mine").c_str());
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
