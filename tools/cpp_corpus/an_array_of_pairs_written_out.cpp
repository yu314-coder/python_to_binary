// `const std::pair<const char *, WORD> values[] = {{"control", VK_CONTROL},
// ...};` and a range-`for` over it. Two things.
//
// An element written as `{a, b}` is what the element is built from. Read as
// an object already built, the copy came out as `values[0] = *&{"control",
// 17}`, which is not an expression in C or anywhere else.
//
// And the extent of the array is read from its declaration, which is how a
// range-`for` over a plain array knows where to stop. The type had to be one
// word for that to be found, so a type written with template arguments was
// not - `values` was taken for a container and asked its `size()`. The count
// is read to the brace that closes the list too, not to the first `}` in the
// text: an element may be a list of its own.
#include <cstdio>
#include <utility>

typedef unsigned short WORD;

struct Named {
    const char *text;
    int n;
    Named(const char *given, int count) { text = given; n = count; }
};

int main() {
    const std::pair<const char *, WORD> values[] = {
        {"control", 17},
        {"shift", 16},
        {"alt", 18}
    };
    int total = 0;
    for (const auto &value : values) total += value.second;

    // A class of the program's own, built the same way.
    Named held[] = {{"one", 1}, {"two", 2}};
    int counted = 0;
    for (const auto &one : held) counted += one.n;

    // And the plain kinds, which have to go on working.
    const int plain[] = {1, 2, 3, 4};
    int summed = 0;
    for (int one : plain) summed += one;

    printf("%s %d|%s %d|%d|%s %s %d|%d\n", values[0].first,
           (int)values[0].second, values[2].first, (int)values[2].second,
           total, held[0].text, held[1].text, counted, summed);
    return 0;
}
