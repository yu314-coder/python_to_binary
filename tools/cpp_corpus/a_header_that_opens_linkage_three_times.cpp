// Three `extern "C"` blocks in a row, two of them empty.
//
// OpenSSL's headers open one after another, and after an include has been
// pasted elsewhere a block can be left empty. The strip took the first out
// and went on searching from past where its brace had been - in text that
// was shorter now - so the block that began right after it was skipped and
// reached the C compiler as `extern "C" {`.
#include <cstdio>

extern "C" {
}
extern "C" {
}
extern "C" {
typedef int linked_int;
static int linked(linked_int a) { return a * 3; }
}
extern "C" int plain(int a);
int plain(int a) { return a + 1; }

int main() {
    printf("%d %d\n", linked(4), plain(4));
    return 0;
}
