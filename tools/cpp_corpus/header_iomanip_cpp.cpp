/* <iomanip>: the three that take an argument. The flag manipulators -
   `std::fixed`, `std::hex`, `std::left` - are not here: they are objects
   rather than calls, and this subset cannot tell which `operator<<` one of
   them means. */
#include <iostream>
#include <iomanip>

int main() {
    std::cout << std::setw(5) << 42 << "|\n";
    std::cout << std::setfill('0') << std::setw(3) << 7 << "\n";
    std::cout << std::setprecision(3) << 1.5 << "\n";
    return 0;
}
