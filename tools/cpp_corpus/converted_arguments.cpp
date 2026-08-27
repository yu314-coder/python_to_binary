#include <cstdio>
#include <string>
static int widthOf(const std::string &text) { return (int)text.size(); }
static int wideWidth(const std::wstring &text) { return (int)text.size(); }
class Panel {
public:
    int title(const std::wstring &text) { return (int)text.size(); }
};
int main() {
    Panel p;
    printf("%d %d %d\n", widthOf("narrow"), wideWidth(L"wide!"), p.title(L"abc"));
    return 0;
}
