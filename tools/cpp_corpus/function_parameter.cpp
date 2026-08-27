#include <cstdio>
#include <string>
#include <functional>
class Panel {
public:
    using Handler = std::function<void(const std::string &)>;
    int show(const char *title, Handler onEvent) {
        onEvent(std::string(title));
        return (int)1;
    }
private:
    int held_;
};
int main() {
    Panel p;
    int n = p.show("hello", [](const std::string &text) {
        printf("event %s\n", text.c_str());
    });
    printf("%d\n", n);
    return 0;
}
