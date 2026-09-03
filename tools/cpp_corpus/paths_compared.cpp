// C++17 compares two paths as paths. py2bin's own <filesystem> gave its path
// only `/`, so `a == b` on two of them reached the C as a comparison of two
// structs - which C refuses - and a program checking whether the file it was
// handed is the one it already holds stopped there. The text is what this
// path is, so the text is what is compared, the way `/` is what joins it.
#include <stdio.h>
#include <string>
#include <filesystem>
namespace fs = std::filesystem;
int main(void) {
    fs::path a("out/dir/leaf.txt");
    fs::path b = fs::path("out/dir") / "leaf.txt";
    fs::path c("other");
    std::wstring wide = a.wstring();
    const wchar_t *forWin32 = wide.c_str();
    fs::path fromWide(wide);
    printf("%d %d %d %d %d %d\n", (int)(a == b), (int)(a != c), (int)(fromWide == a),
           (int)(c < a), (int)(a >= b), (int)(forWin32 != 0));
    return 0;
}
