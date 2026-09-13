/* `if (!sendControlMessage(text.empty() ? "clipboardError" : "clipboardText",
                            text.empty() ? "Windows clipboard is empty." : text))`
   - two conditional arguments to a call taking `const std::string &`, the
   second with a literal in one arm and a string in the other.

   C++ gives that conditional one type, the string, and the literal is
   converted in its own arm. py2bin writes it as the `if` C++ means, in front
   of the statement - and the check for where that may be done read every
   colon as a label, so the first argument's conditional stopped the second
   from being written out, and every keyword as a braceless body, so a call in
   an `if`'s own condition was never lifted. It reached the C stage as a
   `char *` and a `string` in one conditional, refused.

   What may not change is which arm runs: the arm that is not taken calls a
   function that counts, and the count is printed. */
#include <cstdio>
#include <string>

static int made = 0;

static std::string fallback() {
    ++made;
    return "fallback";
}

static bool send(const std::string &kind, const std::string &detail) {
    std::printf("%s|%s\n", kind.c_str(), detail.c_str());
    return kind.size() + detail.size() > 0;
}

int main() {
    std::string text = "copied words";
    std::string none;
    if (!send(text.empty() ? "clipboardError" : "clipboardText",
              text.empty() ? "Windows clipboard is empty." : text)) {
        return 1;
    }
    if (!send(none.empty() ? "clipboardError" : "clipboardText",
              none.empty() ? "Windows clipboard is empty." : none)) {
        return 1;
    }
    int sent = send(text.empty() ? "a" : "b", text.empty() ? fallback() : text);
    std::printf("%d %d\n", sent, made);
    return 0;
}
