// A brace written inside a string literal, which is what a program building
// JSON does on nearly every line. The depth table counted every brace in the
// raw text, so `payload << "}"` read as a scope closing - and from there
// every depth in the file was one too few. A function written after it was
// taken for a nested one, its body was rewritten as somebody else's, and a
// declaration came out after the closing brace of the function that held it.
// Nothing said so: the C simply had a statement where the program never put
// one. The declaration below is `const`, which is the other half of what
// this program watches - the keyword and the type must keep the space
// between them.
#include <cstdio>
#include <string>
#include <sstream>

static std::string described(const std::string &kind, int n) {
    std::ostringstream payload;
    payload << "{\"kind\":\"" << kind << "\",\"n\":" << n << "}";
    const std::string encoded = payload.str();
    return encoded;
}

static int counted(const std::string &text) {
    int braces = 0;
    for (int i = 0; i < (int)text.size(); i++) {
        if (text.at(i) == '}') { braces = braces + 1; }
    }
    return braces;
}

int main() {
    const std::string one = described("chunk", 7);
    const std::string two = described("done", 12);
    printf("%s %s %d %d\n", one.c_str(), two.c_str(), counted(one), counted(two));
    return 0;
}
