/* `typedef int (NAME)(params);` - a typedef of a function type, its name in
   parentheses with no star. OpenSSL's headers write their callbacks this way
   and write some of them twice, a few lines apart. The second time the name
   is a type already, so the parentheses read as a parameter list of one
   type declaring nothing - and the C stage asked for an identifier and found
   the closing parenthesis. A lone name in parentheses is a declarator. */
#include <stdio.h>

typedef int (CHECKER)(const char *text, void *arg);
typedef int (CHECKER)(const char *text, void *arg);
typedef unsigned long (COUNTER)(void);

static int first_letter(const char *text, void *arg) { (void)arg; return text[0]; }
static int last_letter(const char *text, void *arg) {
    int i = 0;
    (void)arg;
    while (text[i] != 0) { i++; }
    return i > 0 ? text[i - 1] : 0;
}
static unsigned long three(void) { return 3UL; }

static int ask(CHECKER *check, const char *text) { return check(text, 0); }

int main(void) {
    CHECKER *held = first_letter;
    COUNTER *how_many = three;
    printf("%d %d %d %lu\n", ask(first_letter, "AB"), ask(last_letter, "AB"),
           held("CD", 0), how_many());
    return 0;
}
