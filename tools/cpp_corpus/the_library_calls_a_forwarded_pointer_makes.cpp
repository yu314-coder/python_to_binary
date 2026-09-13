/* The standard library a program forwarding a mouse and keyboard reaches for:
   `std::clamp` a coordinate into 0..1, `std::round` it onto a 65535 grid,
   `std::abs` a scroll delta against a threshold, `std::transform` a key name
   to lower case, and the clock and thread calls a capture loop paces itself
   with.

   Three of them were wrong in three different ways. `std::clamp` and
   `std::transform` did not exist, which stopped the build. `std::round` was
   refused by name on x86-64, whose one rounding instruction cannot break a tie
   away from zero - also a build that stopped. And `std::abs` on a double built
   and answered wrongly: `std::` is gone by the time a call is compiled, and C
   has one `abs`, taking an int, so the double was truncated on the way in and
   `std::abs(-0.5) > 0.01` was false. A scroll of less than a whole unit was
   ignored, with no diagnostic.

   The rounding cases are the ones a rounding function gets wrong: halves both
   ways, the largest double below one half, a negative number that rounds to
   negative zero, and numbers too large to have a fraction at all. */
#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <string>
#include <thread>

int main() {
    double deltaX = -0.5;
    double tiny = 0.005;
    int a1 = std::abs(deltaX) > 0.01;
    int a2 = std::abs(tiny) > 0.01;
    int a3 = (int)(std::abs(-2.75) * 100.0);
    float af = std::abs(-2.5f);
    long long al = std::abs(-5000000000LL);
    int ai = std::abs(-7);
    std::printf("%d %d %d %.2f %lld %d\n", a1, a2, a3, (double)af, al, ai);

    const double halves[] = {0.5, -0.5, 2.5, -2.5, 0.49999999999999994, -0.3,
                             1e300, 4503599627370497.0, 0.6 * 65535.0};
    for (double v : halves) {
        std::printf("%.17g ", std::round(v));
    }
    std::printf("\n");

    long rounded = static_cast<long>(std::round(0.6 * 65535.0));
    int floored = std::floor(7.9) != 7.9;
    double clamped = std::clamp(deltaX * 120.0, -12000.0, 12000.0);
    int bounded = std::clamp(12, 3, 9);

    std::string name = "Enter";
    std::transform(name.begin(), name.end(), name.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });

    std::string word = "abcdef";
    std::size_t start = 2;
    std::string piece(word.begin() + static_cast<std::ptrdiff_t>(start),
                      word.begin() + static_cast<std::ptrdiff_t>(start + 3));

    char buffer[8] = {0};
    std::memcpy(buffer, "127.0", 5);
    int local = std::strncmp(buffer, "127.", 4) == 0;
    char *end = nullptr;
    double parsed = std::strtod("3.25x", &end);

    auto nextFrame = std::chrono::steady_clock::now();
    nextFrame += std::chrono::milliseconds(1);
    std::this_thread::sleep_until(nextFrame);
    std::thread worker([] {});
    int other = worker.get_id() != std::this_thread::get_id();
    worker.join();

    std::printf("%ld %d %.1f %d %s %s %d %.2f %c %d\n", rounded, floored, clamped,
                bounded, name.c_str(), piece.c_str(), local, parsed, *end, other);
    return 0;
}
