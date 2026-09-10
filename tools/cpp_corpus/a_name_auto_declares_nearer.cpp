// `auto session = std::make_shared<Session>();` here and `BridgeSession
// session(7);` in a function written above it. `auto` is not a type, so the
// reader that finds declarations passes over it - and passed over, the
// declaration further away won: the lambda below captured `session` with the
// type of the other one, so the closure held a `BridgeSession *` and was
// handed the address of a holder.
//
// And `makeResponse(localAddress_, instance, host)` from inside a nested
// block, where `instance` and `host` are declared by the function around it
// and named again in another file's `main`. A block is rewritten on its own,
// so what the function around it declares is in neither the block's text nor
// the file's - and the reader answered with whichever came last.
#include <memory>
#include <string>
#include <vector>
#include <cstdint>
#include <cstdio>

using Bytes = std::vector<std::uint8_t>;

struct Session { int socket = 0; bool closed = false; };
struct Handle { int value = 0; };

class BridgeSession {
public:
    explicit BridgeSession(int mark) : mark_(mark) {}
    void handle(const std::string &message) {
        std::printf("bridge %d %s\n", mark_, message.c_str());
    }
private:
    int mark_;
};

static void handleClient();
static void discovery();
static Bytes makeResponse(const std::string &address, const std::string &instance,
                          const std::string &host);

// First in the file, as a program's entry point usually is.
int main() {
    BridgeSession session(7);
    Handle instance;
    Handle host;
    instance.value = 3;
    host.value = 4;
    auto pass = [&session](const std::string &message) { session.handle(message); };
    pass("hello");
    std::printf("%d %d\n", instance.value, host.value);
    handleClient();
    discovery();
    return 0;
}

static void handleClient() {
    auto session = std::make_shared<Session>();
    session->socket = 45454;
    auto closeSession = [&] {
        session->closed = true;
        std::printf("closed %d\n", session->socket);
    };
    closeSession();
    std::printf("%d\n", (int)session->closed);
}

static void discovery() {
    const std::string instance = "SidecarBridge Windows";
    const std::string host = instance;
    std::string localAddress;
    for (int round = 0; round < 2; round++) {
        if (round == 1) {
            const Bytes response = makeResponse(localAddress, instance, host);
            std::printf("%d\n", (int)response.size());
        }
    }
}

static Bytes makeResponse(const std::string &address, const std::string &instance,
                          const std::string &host) {
    Bytes out;
    out.push_back(static_cast<std::uint8_t>(address.size()));
    out.push_back(static_cast<std::uint8_t>(instance.size()));
    out.push_back(static_cast<std::uint8_t>(host.size()));
    return out;
}
