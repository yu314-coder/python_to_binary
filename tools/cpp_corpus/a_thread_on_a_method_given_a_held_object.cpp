// `thread(&Bridge::captureLoop, this, session)` - a method, the object to
// call it on, and a shared_ptr to hand it. Each thread writes a member of
// its own, so what the program prints does not depend on which runs first.
#include <thread>
#include <memory>
#include <cstdio>

struct Session { int socket = 0; };

// `= delete` on the copy operations, which is how a class holding a thread
// is written. It gives the class an `operator=`, and the trampoline's own
// `Bridge *__py2bin_on = __py2bin_a->on;` was read as an assignment through
// one - so the type stayed standing in front of the call that replaced it.
class Bridge {
public:
    Bridge() {}
    Bridge(const Bridge &) = delete;
    Bridge &operator=(const Bridge &) = delete;

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
