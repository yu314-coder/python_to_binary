// `if (held)` and `static_cast<bool>(held)` on a smart pointer. C++ gives
// every one of these an `operator bool`; py2bin's had only the `!` of it, so
// a condition on a holder was refused - and a cast of one to bool was
// refused too, which is the same question written the other way.
#include <memory>
#include <cstdio>

struct Session { int socket = 0; };

int main() {
    std::shared_ptr<Session> session_;
    std::printf("%d\n", (int)static_cast<bool>(session_));
    if (session_) std::printf("a\n"); else std::printf("b\n");

    session_ = std::make_shared<Session>();
    session_->socket = 45454;
    std::printf("%d\n", (int)static_cast<bool>(session_));
    if (session_) std::printf("c %d\n", session_->socket);

    bool busy = false;
    busy = static_cast<bool>(session_);
    std::printf("%d\n", (int)busy);

    std::unique_ptr<Session> owned;
    if (!owned) std::printf("empty\n");
    owned.reset(new Session());
    if (owned) std::printf("owned\n");
    std::printf("%d\n", (int)static_cast<bool>(owned));
    return 0;
}
