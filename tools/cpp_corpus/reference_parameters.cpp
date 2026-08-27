#include <cstdio>
#include <string>
class Store {
public:
    bool load(const std::wstring& title, const std::string& body);
    int sizes() { return (int)title_.size() * 10 + (int)body_.size(); }
private:
    std::wstring title_;
    std::string body_;
};
bool Store::load(const std::wstring& title, const std::string& body) {
    title_ = title;
    body_ = body;
    return true;
}
int main() {
    Store s;
    std::wstring t;
    t.assign(L"abcd");
    std::string b("xyz");
    s.load(t, b);
    printf("%d\n", s.sizes());
    return 0;
}
