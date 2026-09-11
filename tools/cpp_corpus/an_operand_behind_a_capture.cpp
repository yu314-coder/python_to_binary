// `if (session_ == session) { session_.reset(); }` inside a closure. The
// capture is a pointer, so the right operand arrives as `(*this->session)`
// - not a name, so the rule that hands an object over by address did not
// fire and the operator was given one by value. `&(*p)` is `p`.
#include <memory>
#include <mutex>
#include <cstdio>

struct Session { int socket = 0; };

class Holder {
public:
    std::shared_ptr<Session> session_;
    std::mutex sessionMutex_;
    void handle();
};

void Holder::handle() {
    auto session = std::make_shared<Session>();
    session->socket = 45454;
    session_ = session;
    auto closeSession = [&] {
        std::lock_guard<std::mutex> lock(sessionMutex_);
        if (session_ == session) { session_.reset(); }
        std::printf("closed %d\n", session->socket);
    };
    closeSession();
    std::printf("%d\n", (int)static_cast<bool>(session_));
}

int main() { Holder h; h.handle(); return 0; }
