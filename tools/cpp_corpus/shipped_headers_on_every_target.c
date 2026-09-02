/* Two of the headers py2bin ships compiled on five targets and not the
   sixth, and only a build for that one target said so.

   <sys/types.h> wrote `typedef long ssize_t;` while the compiler settles
   ssize_t from the target's data model. On the LP64 targets the two answers
   agree by luck; on Windows a `long` is four bytes and the model says eight,
   so the header stopped compiling there with "'ssize_t' is already a
   different type".

   <objidl.h> names the handles STGMEDIUM chooses between. Three of them it
   declares itself, so as not to need <windows.h> - which refuses a target
   that is not Windows - and HBITMAP and HGLOBAL were left out of that list.
   So it compiled on Windows and nowhere else, which is the opposite of what
   these COM headers are for: declaring an interface once and building it for
   six machines.

   Built for every target by `cpp_sweep.sh targets`, which is the only thing
   that would have caught either. */
#include <sys/types.h>
#include <objidl.h>
#include <stdio.h>

int main(void) {
    ssize_t signed_width = -3;
    STGMEDIUM medium;
    medium.tymed = TYMED_HGLOBAL;
    medium.hGlobal = 0;
    medium.pUnkForRelease = 0;
    /* ssize_t is as wide as a pointer on every target py2bin has, which is
       the fact the two spellings disagreed about. */
    printf("%d %d %d\n",
           (int)(sizeof(ssize_t) == sizeof(void *)),
           (int)(signed_width < 0),
           (int)(medium.tymed == TYMED_HGLOBAL));
    return 0;
}
