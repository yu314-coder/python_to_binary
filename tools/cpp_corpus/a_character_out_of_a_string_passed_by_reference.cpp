// `std::string formatCode(const std::string &digits) { ... result +=
// digits[i]; }` - a container passed the way every container is passed, and
// one of its elements appended.
//
// A reference is a pointer here, and `digits[i]` on a pointer reads as
// pointer arithmetic - so the type of `digits[i]` came back as `string`, and
// `result += digits[i]` chose the overload that appends a whole string over
// the one that appends a character. The pass that *writes* the subscript out
// already reads it as the container's own operator; this is what it took for
// the deduction to agree with it.
//
// The class body is not in the text by the time the question is asked, so
// what the operator answers is read off the prototype emitted above every
// call - where the `&` of a reference return has already become a `*`.
#include <cstdio>
#include <string>
#include <vector>

static std::string formatted(const std::string &digits) {
    std::string result;
    for (size_t i = 0; i < digits.size(); ++i) {
        if (i != 0 && i % 4 == 0) result += '-';
        result += digits[i];
    }
    return result;
}

static int total(const std::vector<int> &given) {
    int sum = 0;
    for (size_t i = 0; i < given.size(); ++i) sum += given[i];
    return sum;
}

// And by value, which has to go on meaning the same thing.
static std::string firstTwo(std::string given) {
    std::string out;
    out += given[0];
    out += given[1];
    return out;
}

int main() {
    std::vector<int> numbers;
    numbers.push_back(3);
    numbers.push_back(4);
    printf("%s|%d|%s\n", formatted("123456789").c_str(), total(numbers),
           firstTwo("xyz").c_str());
    return 0;
}
