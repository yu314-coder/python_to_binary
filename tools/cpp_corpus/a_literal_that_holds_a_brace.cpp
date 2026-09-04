// A string literal holding a brace, a semicolon or a quote, wherever it goes.
//
// `std::string text = "{\"k\": 1}";` is how a program that speaks JSON writes
// one. The pattern reading `T name(args);` matched the raw text and stopped
// at the brace, so the declaration reached the C as C++; the class-body
// reader cut a member at the brace inside its initialiser and handed the
// rest to the method reader; `string s = raw;` from a `const char *` was
// left as a struct assigned a pointer; and `"<" + key` - a literal on the
// left of an object - had no operator to go to.
#include <stdio.h>
#include <string>

struct Message {
    std::string body = "{\"k\": 1}";
    std::string tail = "a;b";
    std::string wrap(const std::string &key) { return "\"" + key + "\": " + body; }
};

int main(void) {
    std::string open = "{";
    std::string close("}");
    std::string both = "x;y";
    std::string json = "{\"k\": 1}";
    std::string quoted("say \"hi\" \\ done");
    printf("%s %s %s %s %s\n", open.c_str(), close.c_str(), both.c_str(), json.c_str(), quoted.c_str());
    const char *raw = "plain";
    std::string s = raw;
    std::string t(raw);
    std::string key = "k";
    std::string quotedKey = "\"" + key + "\"";
    std::string angled = "<" + key + ">";
    printf("%s %s %s %s %d\n", s.c_str(), t.c_str(), quotedKey.c_str(), angled.c_str(), (int)quotedKey.size());
    Message m;
    printf("%s %s %s\n", m.body.c_str(), m.tail.c_str(), m.wrap("id").c_str());
    return 0;
}
