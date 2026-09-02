/* <functional>: a std::function holding a free function, then a lambda that
   captured something. */
#include <functional>
#include <cstdio>

static int twice(int value) { return value * 2; }

int main() {
    std::function<int(int)> held = twice;
    printf("%d\n", held(21));
    int captured = 5;
    std::function<int(int)> lambda =
        [captured](int value) { return value + captured; };
    printf("%d\n", lambda(1));
    return 0;
}
