// `if (session_ == session) { session_.reset(); }` - two holders compared
// against each other, and one let go of without anything to put in its
// place. The shipped holders compared only against a raw pointer and had
// no `reset` that takes nothing.
#include <memory>
#include <cstdio>

struct Session { int socket = 0; };

class Holder {
public:
    std::shared_ptr<Session> session_;
    void close(const std::shared_ptr<Session> &session) {
        if (session_ == session) { session_.reset(); }
    }
};

int main() {
    Holder holder;
    auto one = std::make_shared<Session>();
    auto two = std::make_shared<Session>();
    one->socket = 1;
    two->socket = 2;

    holder.session_ = one;
    holder.close(two);
    std::printf("%d\n", (int)static_cast<bool>(holder.session_));
    holder.close(one);
    std::printf("%d\n", (int)static_cast<bool>(holder.session_));

    std::printf("%d %d\n", (int)(one == one), (int)(one != two));

    std::unique_ptr<Session> owned(new Session());
    owned.reset();
    std::printf("%d\n", (int)static_cast<bool>(owned));

    // And against nothing at all, which C++ reaches by turning the null
    // constant into a pointer rather than into a holder.
    std::printf("%d %d\n", (int)(owned == nullptr), (int)(one != nullptr));
    return 0;
}
