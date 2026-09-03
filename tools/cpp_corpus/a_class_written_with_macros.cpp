// The C++ stage runs in front of the preprocessor, so every `#define` below
// was still its own name while these classes were being taken apart - and
// read as ordinary words the names went wrong in different ways. `FIELDS`
// standing where a member goes was swallowed into the return type of the
// method after it, which took both of the ints it declares out of the struct:
// `sizeof` came out eight bytes short and the program still ran. `VIRT` hid
// the one word that gives `who` a slot in the vtable, so a base pointer
// called the base's copy of it. `BASE` named a class the translator said this
// translation unit does not declare, and `GETTER` was a method nothing could
// see. What each one stands for is read off the directive itself here, which
// is the only thing there is to read before the preprocessor has had the file.
#include <stdio.h>

#define ELEM int
#define N 4
#define FIELDS int a; int b;
#define PUB public:
#define VIRT virtual
#define GETTER int doubled() { return v * 2; }
#define BASE Animal
#define COUNTED Counter

struct Counter {
    int made;
    Counter() { made = 1; }
};

class Boxed {
    int hidden;
PUB
    FIELDS
    ELEM room[N];
    COUNTED counted;
    int v;
    GETTER
    Boxed() {
        hidden = 3;
        a = 1;
        b = 2;
        v = 6;
        for (int i = 0; i < N; i++) room[i] = i;
    }
    int total() {
        int sum = hidden + a + b;
        for (int i = 0; i < N; i++) sum += room[i];
        return sum + counted.made;
    }
};

struct Animal {
    int legs;
    Animal() { legs = 4; }
    VIRT int who() { return 1; }
};

struct Dog : BASE {
    int who() { return 2; }
};

int main(void) {
    Boxed boxed;
    Dog dog;
    Animal *seen = &dog;
    printf("total %d doubled %d\n", boxed.total(), boxed.doubled());
    printf("size %d room %d\n", (int)sizeof(Boxed), boxed.room[3]);
    printf("legs %d who %d\n", dog.legs, seen->who());
    return 0;
}
