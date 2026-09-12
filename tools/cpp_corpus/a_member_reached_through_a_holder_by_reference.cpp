// `sendFrame(session->socket, frame, session->sendMutex)` where the third
// parameter is a `std::mutex&` and `session` is a smart pointer. Taking the
// address of a reference argument is only done where the argument's type
// can be read, and reading a member through a holder means asking the
// holder what its `operator->` answers - which is a method, and by the time
// a body is rewritten the holder's own body is no longer in the text.
#include <memory>
#include <mutex>
#include <vector>
#include <cstdio>

struct Session {
    int socket = 0;
    std::mutex sendMutex;
};

static bool sendFrame(int socket, const std::vector<int> &frame, std::mutex &lock) {
    std::lock_guard<std::mutex> held(lock);
    return socket >= 0 && !frame.empty();
}

int main() {
    auto session = std::make_shared<Session>();
    session->socket = 3;
    std::vector<int> frame;
    frame.push_back(1);
    std::printf("%d\n", (int)sendFrame(session->socket, frame, session->sendMutex));
    return 0;
}
