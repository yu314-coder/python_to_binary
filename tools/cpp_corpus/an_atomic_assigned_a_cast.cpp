// `held_ = static_cast<uintptr_t>(value);` where `held_` is an atomic. The
// assignment operator is written for a name and for a number, and a cast is
// neither by the time the rewrite reads it - so the statement was left as it
// stood and the C stage was handed a struct being assigned an integer.
#include <atomic>
#include <cstdint>
#include <cstdio>

class T {
public:
    std::atomic<std::uintptr_t> held_{0};
    std::atomic<bool> flag_{false};
    void set(unsigned long long value);
};

void T::set(unsigned long long value) {
    held_ = static_cast<std::uintptr_t>(value);
    flag_ = true;
}

int main() {
    T t;
    t.set(45454);
    std::printf("%llu %d\n", (unsigned long long)t.held_.load(), (int)t.flag_.load());
    t.held_ = 0;
    std::printf("%llu\n", (unsigned long long)t.held_.load());
    t.held_ = (std::uintptr_t)(7 + 1);
    std::printf("%llu\n", (unsigned long long)t.held_.load());
    return 0;
}
