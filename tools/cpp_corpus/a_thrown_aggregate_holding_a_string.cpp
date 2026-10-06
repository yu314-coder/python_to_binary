// `throw ParseError{"unexpected token at " + std::to_string(line), line};` - an aggregate with a string member, built from braces.
#include <cstdio>
#include <string>
struct ParseError { std::string message; int line; };
void parse(int line) { if (line == 3) throw ParseError{"unexpected token at " + std::to_string(line), line}; }
int main() {
    try { for (int i = 1; i < 5; i++) parse(i); } catch (const ParseError &e) { printf("%s (%d)\n", e.message.c_str(), e.line); }
    return 0;
}
