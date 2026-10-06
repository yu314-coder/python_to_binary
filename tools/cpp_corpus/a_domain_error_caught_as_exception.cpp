// `std::domain_error` caught as `std::exception`, inside a loop that records each answer.
#include <cstdio>
#include <stdexcept>
int divide(int a, int b) { if (b == 0) throw std::domain_error("divide by zero"); return a / b; }
int main() {
    int results[3]; int bs[3] = {2, 0, 5};
    for (int i = 0; i < 3; i++) {
        try { results[i] = divide(10, bs[i]); }
        catch (const std::exception &e) { printf("%s\n", e.what()); results[i] = -1; }
    }
    printf("%d %d %d\n", results[0], results[1], results[2]);
    return 0;
}
