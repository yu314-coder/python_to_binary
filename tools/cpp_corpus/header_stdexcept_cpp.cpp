/* <stdexcept>: two of the standard ones, one caught by its own type and one
   through the base every exception has. */
#include <stdexcept>
#include <cstdio>

int main() {
    try {
        throw std::runtime_error("gone wrong");
    } catch (const std::runtime_error &caught) {
        printf("%s\n", caught.what());
    }
    try {
        throw std::out_of_range("past the end");
    } catch (const std::exception &caught) {
        printf("%s\n", caught.what());
    }
    return 0;
}
