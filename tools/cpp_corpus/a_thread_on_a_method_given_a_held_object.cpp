// `thread(&Bridge::captureLoop, this, session)` - a method, the object to
// call it on, and a shared_ptr to hand it. Each thread writes a member of
// its own, so what the program prints does not depend on which runs first.
#include <thread>
#include <memory>
#include <cstdio>

struct Session { int socket = 0; };

class Bridge {
public:
    std::thread worker_;
    std::thread captureWorker_;
    int fromRun = 0;
    int fromCapture = 0;
    void run();
    void captureLoop(std::shared_ptr<Session> session);
    void start();
};

void Bridge::run() { fromRun = 1; }

void Bridge::captureLoop(std::shared_ptr<Session> session) {
    fromCapture = session->socket;
}

void Bridge::start() {
    worker_ = std::thread(&Bridge::run, this);
    auto session = std::make_shared<Session>();
    session->socket = 41;
    captureWorker_ = std::thread(&Bridge::captureLoop, this, session);
    worker_.join();
    captureWorker_.join();
}

int main() {
    Bridge b;
    b.start();
    std::printf("%d %d\n", b.fromRun, b.fromCapture);
    return 0;
}
