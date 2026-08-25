#include <stdio.h>
#include <string>
class Bridge {
public:
    std::string title; int events;
    Bridge();
    void on(const char *what);
    int drive();
    const char *html() const { return "<div style=\"a{b}\">{ \"k\": 1 }</div>"; }
};
Bridge::Bridge() { title.assign("wv"); events = 0; }
void Bridge::on(const char *what) { events = events + 1; printf("%s:%s ", title.c_str(), what); }
int Bridge::drive() {
    auto handler = [this](const char *what) { on(what); };
    handler("ready");
    handler("close");
    return events;
}
int main() { Bridge b; printf("| %d %d\n", b.drive(), (int)(b.html()[0] == '<')); return 0; }
