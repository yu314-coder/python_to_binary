/* <new>: what the header is included for even where nothing in it is named -
   `new` and `delete` themselves, including the array forms. */
#include <new>
#include <cstdio>

int main() {
    int *one = new int(9);
    printf("%d\n", *one);
    delete one;
    int *many = new int[4];
    for (int i = 0; i < 4; i++) { many[i] = i * 3; }
    printf("%d %d\n", many[1], many[3]);
    delete[] many;
    return 0;
}
