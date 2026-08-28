#include <iostream>
#include <iomanip>
int main() {
    std::cout << 1.5 << " " << 3.14159 << std::endl;
    std::cout << std::setprecision(3) << 3.14159 << std::endl;
    std::cout << std::setw(6) << std::setfill('.') << 42 << std::endl;
    return 0;
}
