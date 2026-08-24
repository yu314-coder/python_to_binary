#include <stdio.h>
#include <stdexcept>
int risky(int n) {
    if (n < 0) throw std::runtime_error("bad input");
    if (n == 0) throw std::out_of_range("zero");
    return n;
}
int main(void) {
    try { risky(-1); } catch (std::exception &e) { printf("A:%s|", e.what()); }
    try { risky(0); } catch (std::out_of_range e) { printf("C:%s|", e.what()); }
    printf("%d\n", risky(4));
    return 0;
}
