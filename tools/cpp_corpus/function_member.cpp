#include <cstdio>
#include <string>
#include <functional>
class Session {
public:
    using EventHandler = std::function<void(const std::string &)>;
    explicit Session(EventHandler eventHandler);
    void fire(const char *what);
private:
    EventHandler handler_;
};
Session::Session(EventHandler eventHandler)
    : handler_(std::move(eventHandler)) { }
void Session::fire(const char *what) {
    if (!handler_) { return; }
    handler_(std::string(what));
}
int main() {
    Session s([](const std::string &text) { printf("got %s\n", text.c_str()); });
    s.fire("ping");
    return 0;
}
