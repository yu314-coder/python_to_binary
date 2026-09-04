// A local declared above a block, used inside it by a call that has to choose.
//
// `text.find(quotedKey)` inside an `if` chooses between `find(const char *)`,
// `find(char)` and `find(string)` by the type of `quotedKey` - declared in the
// body around the block. The block was handed that body's objects by name
// but not its text, and the reader that types an argument reads text, so
// the choice could not be made and the program was refused with "cannot
// tell which is meant by `quotedKey`". In a function, in a method, in an
// `if`, a `for` and a `while`, a `const` local, and a block-local of the same
// name that shadows the outer one.
#include <stdio.h>
#include <string>

struct Parser {
    std::string text = "{\"k\": 1, \"v\": 2}";
    int look(const std::string &key) {
        std::string quotedKey = "\"" + key + "\"";
        if (!text.empty()) {
            return (int)text.find(quotedKey);
        }
        return -1;
    }
    int count(const std::string &key) {
        std::string needle = "\"" + key + "\"";
        int found = 0;
        for (int i = 0; i < 3; ++i) {
            if ((int)text.find(needle) >= 0) found += 1;
        }
        return found;
    }
};

int main(void) {
    std::string text = "{\"k\": 1, \"v\": 2}";
    std::string quotedKey = "\"v\"";
    const std::string fixed = "\"k\"";
    int at = -1;
    if (text.size() > 0) {
        at = (int)text.find(quotedKey);
    }
    int i = 0;
    int last = -1;
    while (i < 2) {
        last = (int)text.find(fixed);
        ++i;
    }
    int shadowed = -1;
    {
        std::string quotedKey = "\"k\"";
        shadowed = (int)text.find(quotedKey);
    }
    Parser p;
    printf("%d %d %d %d %d\n", at, last, shadowed, p.look("v"), p.count("k"));
    return 0;
}
