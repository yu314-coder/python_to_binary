// A raw string literal is the text between its parentheses, quotes,
// backslashes, macro names and line breaks included. Read as an ordinary
// literal it was two strings with code between them, and the C stage saw a
// string where it expected a comma.
#include <stdio.h>

#define GREETING R"(say "hi")"

int main(void) {
    const char *plain = R"(a "b" __LINE__ c\n)";
    const char *two_lines = R"(first
second)";
    const char *delimited = R"xy(has )" inside)xy";
    printf("%s\n", plain);
    printf("%s\n", two_lines);
    printf("%s\n", delimited);
    printf("%s\n", GREETING);
    printf("%d\n", (int)sizeof(R"(abc)"));
    printf("%d\n", __LINE__);
    return 0;
}
