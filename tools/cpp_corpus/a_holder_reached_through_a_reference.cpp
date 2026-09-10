// A lambda that captures a smart pointer and reaches through it. Inside the
// closure the use is written through the capture, so the arrow stands on
// `(*this->session)` and not on a name at all - and the pass that turns an
// arrow into the holder's own operator keys on names.
//
// And `describe(value)`, where the parameter is a `const std::string &`: the
// reader was asked what `value` is without being told where the call is, so
// the `int value` in the function below answered for the string here and the
// address the parameter wants was not taken.
#include <memory>
#include <string>
#include <cstdio>

struct Session { int socket = 0; bool closed = false; };

class BridgeSession {
public:
    explicit BridgeSession(int mark) : mark_(mark) {}
    void handle(const std::string &message) {
        std::printf("bridge %d %s\n", mark_, message.c_str());
    }
private:
    int mark_;
};

static void describe(const std::string &value) {
    std::printf("describe %s\n", value.c_str());
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

    std::string value = "held";
    describe(value);
}

// A later function that gives the same name to something else.
static int doubled() {
    int value = 21;
    return value * 2;
}

int main() {
    BridgeSession session(7);
    auto pass = [&session](const std::string &message) { session.handle(message); };
    pass("hello");
    handleClient();
    std::printf("%d\n", doubled());
    return 0;
}
