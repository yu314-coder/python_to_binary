/* <string.h>: the copies, the comparisons and the searches. Every one of
   these is compiled from the C in py2bin's own header. */
#include <string.h>
#include <stdio.h>

int main(void) {
    char room[32];
    const char *held = "abcabc";
    strcpy(room, "hello");
    strcat(room, ", world");
    printf("%s %d\n", room, (int)strlen(room));
    printf("%d %d\n", strcmp("abc", "abd") < 0, strncmp("abc", "abd", 2));
    printf("%d %d\n", (int)(strchr(held, 'b') - held),
           (int)(strrchr(held, 'b') - held));
    printf("%d\n", (int)(strstr(held, "cab") - held));
    strncpy(room, "xy", 2);
    printf("%s\n", room);
    memset(room, 'z', 3);
    printf("%s\n", room);
    printf("%d\n", memcmp("ab", "ab", 2));
    return 0;
}
