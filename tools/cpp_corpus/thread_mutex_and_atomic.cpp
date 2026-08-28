#include <cstdio>
#include <thread>
#include <mutex>
#include <atomic>

struct Worker {
    std::atomic<bool> running_;
    std::mutex guard_;
    long counted_;
    std::thread a_;
    std::thread b_;
    Worker() { running_.store(false); counted_ = 0; }
    void loop() {
        for (int i = 0; i < 20000; i++) {
            std::lock_guard hold(guard_);
            counted_ += 1;
        }
    }
    void start() {
        running_.store(true);
        a_ = std::thread(&Worker::loop, this);
        b_ = std::thread(&Worker::loop, this);
    }
    void stop() {
        if (a_.joinable()) a_.join();
        if (b_.joinable()) b_.join();
        running_.store(false);
    }
};

int main() {
    Worker w;
    w.start();
    w.stop();
    printf("%ld %d\n", w.counted_, (int)w.running_.load());
    return 0;
}
