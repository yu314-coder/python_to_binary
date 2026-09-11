// `session->closed = true;` inside a lambda that captured `session`. The
// capture is a pointer to the holder, so the left of the assignment arrives
// as `(*this->session)->closed` - not a name, and the pass that turns an
// assignment into the class's own `operator=` walks the names this scope
// knows. The atomic member was handed an int.
#include <atomic>
#include <memory>
#include <cstdio>

struct Session {
    int socket = 0;
    std::atomic<bool> authenticated{false};
    std::atomic<bool> closed{false};
    std::atomic<bool> socketClosed{false};
};

int main() {
    auto session = std::make_shared<Session>();
    session->socket = 45454;
    auto closeSession = [&] {
        session->closed = true;
        session->authenticated = false;
        // And a call on such a member, which is the same receiver written
        // with a method after it rather than an `=`.
        if (!session->socketClosed.exchange(true)) {
            std::printf("closing %d\n", session->socket);
        }
        std::printf("again %d\n", (int)session->socketClosed.exchange(true));
    };
    closeSession();
    closeSession();
    std::printf("%d %d %d\n", (int)session->closed.load(),
                (int)session->authenticated.load(),
                (int)session->socketClosed.load());
    return 0;
}
