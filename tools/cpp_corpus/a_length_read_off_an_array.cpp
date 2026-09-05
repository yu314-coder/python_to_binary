// `std::size(buffer)` on a plain array: a function template taking the array
// by reference with its bound as a template parameter, which is how the
// standard library asks for a length. py2bin read the parameter list as one
// it had never seen a function in, so `<iterator>` could not be written at
// all - and a program asking a platform for a name into a fixed buffer is
// ordinary. The container form is beside it, and an array must not reach it.
#include <cstdio>
#include <iterator>
#include <string>
#include <vector>

int main() {
    wchar_t buffer[256];
    int numbers[7] = {1, 2, 3, 4, 5, 6, 7};
    char tag[3] = {'a', 'b', 0};
    std::vector<int> held;
    held.push_back(10);
    held.push_back(20);
    std::string text = "hello";
    printf("%d %d %d %d %d\n", (int)std::size(buffer), (int)std::size(numbers),
           (int)std::size(tag), (int)std::size(held), (int)std::size(text));
    printf("%d %d %d\n", (int)std::empty(held), *std::begin(numbers),
           *std::data(numbers));
    int *first = std::begin(numbers);
    int *last = std::end(numbers);
    std::advance(first, 2);
    printf("%ld %d %d\n", std::distance(std::begin(numbers), last), *first,
           *std::prev(last));
    return 0;
}
