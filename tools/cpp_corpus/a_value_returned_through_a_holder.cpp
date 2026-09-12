// `sendFrame(socket, session->encrypted(packet), session->sendMutex)` - a
// method that answers an object by value, reached through a smart pointer.
// A value return writes into space the caller provides, so the call needs
// a temporary; the pass that makes one looks the method up on the holder,
// which does not have it, and the pass that fills the temporary ran after
// the arrow had already become a call on what it answers.
#include <memory>
#include <mutex>
#include <vector>
#include <cstdint>
#include <cstdio>

using Bytes = std::vector<std::uint8_t>;

struct Session {
    int socket = 0;
    std::mutex sendMutex;
    Bytes encrypted(const Bytes &packet) {
        Bytes out;
        for (std::size_t i = 0; i < packet.size(); i++) out.push_back(packet[i] ^ 1);
        return out;
    }
};

static bool sendFrame(int socket, const Bytes &frame, std::mutex &lock) {
    std::lock_guard<std::mutex> held(lock);
    return socket >= 0 && !frame.empty();
}

int main() {
    auto session = std::make_shared<Session>();
    session->socket = 3;
    Bytes packet;
    packet.push_back(4);
    std::printf("%d\n", (int)sendFrame(session->socket,
                                       session->encrypted(packet),
                                       session->sendMutex));
    return 0;
}
