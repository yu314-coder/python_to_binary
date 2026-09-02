/* <filesystem>: a path taken apart. Nothing here touches a disk, so the
   answer is the same on every machine. */
#include <filesystem>
#include <cstdio>

int main() {
    std::filesystem::path where("/one/two/three.txt");
    printf("%s\n", where.filename().string().c_str());
    printf("%s\n", where.extension().string().c_str());
    printf("%s\n", where.parent_path().string().c_str());
    printf("%s\n", where.stem().string().c_str());
    return 0;
}
