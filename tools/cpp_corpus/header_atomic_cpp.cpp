/* <atomic>: loaded, stored and added to. The operator forms - `c++` and
   `c += 1` - are not here; the named ones are what this subset has. */
#include <atomic>
#include <cstdio>

int main() {
    std::atomic<int> counter(0);
    counter.fetch_add(3);
    counter.fetch_add(4);
    printf("%d\n", counter.load());
    counter.store(10);
    printf("%d\n", counter.load());
    std::atomic<bool> flag(false);
    flag.store(true);
    printf("%d\n", (int)flag.load());
    return 0;
}
