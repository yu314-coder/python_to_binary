// `path directory(buffer);` inside an `if`, where `buffer` is declared by
// the function around it and another function declares that name with the
// other width. A block is rewritten on its own, so the block's text does
// not hold the declaration - and the reader fell back to the FIRST one
// anywhere, which chose the `char *` constructor for a `wchar_t *`.
#include <filesystem>
#include <cstdio>

// A `buffer` of the other width, written first, as an earlier function in a
// program of several files would have it.
static void narrow() {
    char buffer[8] = "abc";
    std::filesystem::path here(buffer);
    std::printf("%s\n", here.string().c_str());
}

static void wide(int length) {
    wchar_t buffer[8] = L"wxyz";
    if (length > 0) {
        std::filesystem::path directory(buffer);
        std::printf("%d\n", (int)directory.string().size());
    }
}

int main() { narrow(); wide(1); return 0; }
