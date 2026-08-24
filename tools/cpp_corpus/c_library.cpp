#include <stdio.h>
#include <string.h>
#include <ctype.h>
int main(void) {
    char buf[32];
    strcpy(buf, "Hello ");
    strcat(buf, "World");
    printf("%s|%lu|%d|", buf, strlen(buf), strcmp("a", "b"));
    printf("%s|%d|", strstr(buf, "o W"), (int)(strchr(buf, 'W') - buf));
    char copy[8];
    memset(copy, 0, 8);
    memcpy(copy, "abc", 3);
    printf("%s|%d|", copy, memcmp("ab", "ab", 2));
    printf("%d%d%d|%c%c\n", isdigit('7'), isalpha('z'), isspace('\t'),
           tolower('X'), toupper('x'));
    return 0;
}
