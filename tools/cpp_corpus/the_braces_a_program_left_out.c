/* C lets the inner braces go. `struct S a = {1, 2, 3};` where the first
   member is an array of two fills that array and then the member after it -
   a value standing where an aggregate goes is not the aggregate, it is the
   first thing inside it, and as many values as it needs are taken. py2bin
   counted one value per member and stopped, which refused a form every C
   compiler takes; and it is how `std::array` reaches the C stage, since that
   is a struct whose one member is an array. The braces are put back before
   anything reads the list, so everything below still sees one value per
   member. */
#include <stdio.h>

struct Inner {
    int a;
    int b;
};

struct S {
    int items[3];
    struct Inner in;
    char name[4];
};

struct Rows {
    int cell[2][3];
};

static struct S global = {1, 2, 3, 4, 5, "ab"};
static int grid[2][2] = {1, 2, 3, 4};
static struct Inner many[2] = {1, 2, 3, 4};

int main(void) {
    struct S all = {1, 2, 3, 4, 5, "hi"};
    struct S some = {1, 2};
    /* Written out in full, which has to keep meaning what it did. */
    struct S written = {{1, 2, 3}, {4, 5}, "ab"};
    struct Rows rows = {1, 2, 3, 4, 5, 6};
    struct Rows nested = {{{1, 2, 3}, {4, 5, 6}}};
    int flat[2][2] = {1, 2, 3, 4};

    printf("%d %d %d %d %d %s\n", all.items[0], all.items[2], all.in.a,
           all.in.b, all.name[1], all.name);
    printf("%d %d %d %d\n", some.items[0], some.items[1], some.items[2],
           some.in.a);
    printf("%d %d %s\n", written.items[2], written.in.b, written.name);
    printf("%d %d %d %d\n", rows.cell[0][0], rows.cell[1][0], nested.cell[1][2],
           flat[1][0]);
    printf("%d %d %d %d %s\n", global.items[0], global.items[2], global.in.a,
           global.in.b, global.name);
    printf("%d %d %d %d %d %d\n", grid[0][1], grid[1][0], many[0].a, many[0].b,
           many[1].a, many[1].b);
    return 0;
}
