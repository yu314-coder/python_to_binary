#include <stdio.h>
#include <filesystem>
namespace fs = std::filesystem;
int main(void) {
    fs::path f("/tmp/py2bin-cmp/deep/note.tar.gz");
    printf("%s|%s|%s|%s\n",
           f.filename().c_str(), f.stem().c_str(),
           f.extension().c_str(), f.parent_path().c_str());
    fs::path d("/tmp/py2bin-cmp");
    printf("exists=%d ", (int)fs::exists(d));
    fs::create_directory(d);
    printf("made=%d dir=%d ", (int)fs::exists(d), (int)fs::is_directory(d));
    fs::remove(d);
    printf("gone=%d\n", (int)fs::exists(d));
    return 0;
}
