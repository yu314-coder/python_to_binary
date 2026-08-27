#include <cstdio>
#include <string>
class Holder {
public:
    std::wstring text;
    Holder() { text.assign(L"held"); }
    std::wstring copy() { return text; }
};
static int widthOf(const std::wstring &t) { return (int)t.size(); }
class Sink {
public:
    int take(const std::wstring &t) { return (int)t.size() * 2; }
};
int main() {
    Holder h;
    Sink s;
    printf("%d %d\n", widthOf(h.copy()), s.take(h.copy()));
    return 0;
}
