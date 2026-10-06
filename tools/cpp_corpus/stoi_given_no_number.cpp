// `std::stoi("x9")` throws `std::invalid_argument`, caught inside a loop that goes on.
#include <cstdio>
#include <stdexcept>
#include <string>
int main() {
    int total = 0;
    const char *inputs[] = {"12", "x9", "7"};
    for (const char *in : inputs) {
        try { total += std::stoi(in); }
        catch (const std::invalid_argument &e) { total += 1000; }
    }
    printf("%d\n", total);
    return 0;
}
