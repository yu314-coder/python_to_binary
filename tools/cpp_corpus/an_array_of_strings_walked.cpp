// An array of strings walked with a range-for, the name written against the
// star and the list ending in a comma. Neither was read: the walk asked the
// array for its size() as if it were a container, and the comma counted as
// one more element, past the end.
#include <cstdio>
int main() {
    const char *shapes[] = { "a", "b", "c", };
    for (const char *s : shapes) printf("%s\n", s);
    int nums[] = {1, 2, 3}; int t = 0;
    for (int n : nums) t += n;
    printf("%d\n", t);
    return 0;
}
