// `for (wchar_t character : wideFromUtf8(text))` - walking over what a call
// answered, which is how a program handles a string it has just converted.
// Three things stood between py2bin and it.
//
// The range was read as "everything up to the first `)`", so a range holding
// parentheses of its own matched nothing and the loop reached the C stage
// still written in C++. No pattern counts parentheses, so the header is
// scanned to the `)` that closes the `for` - a call inside a call is two
// deep, and a pattern written for one level would have missed it.
//
// The header was read one code piece at a time, and a literal in the range
// splits it into two - so `widened("abc")` was half a header either side of
// the string. Read over the whole text with the literals blanked, it is one
// match.
//
// And the call is made once. A range-`for` becomes an index loop asking the
// range its size and then indexing it, so a call written there would have
// been made once for the size and once per element - which is not what the
// program says.
#include <cstdio>
#include <string>
#include <vector>

static int made = 0;

static std::wstring widened(const std::string &given) {
    made = made + 1;
    std::wstring out;
    for (unsigned long i = 0; i < given.size(); i++) {
        out.push_back((wchar_t)given.at(i));
    }
    return out;
}

static std::vector<int> counted(int upto) {
    made = made + 1;
    std::vector<int> out;
    for (int i = 1; i <= upto; i++) out.push_back(i);
    return out;
}

int main() {
    int total = 0;
    for (wchar_t character : widened("abc")) total += (int)character;

    int summed = 0;
    for (int one : counted(4)) summed += one;

    // A call inside a call, which is where a pattern counting one level of
    // parentheses stops.
    int nested = 0;
    for (wchar_t character : widened(std::string("de"))) nested += (int)character;

    // The plain kinds, which have to go on working.
    std::vector<int> plain;
    plain.push_back(5);
    plain.push_back(6);
    int again = 0;
    for (int one : plain) again += one;

    printf("%d %d %d %d %d\n", total, summed, again, nested, made);
    return 0;
}
