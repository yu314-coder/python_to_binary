// A struct typedef whose member carries a decoration, read as the data it is.
//
// `typedef struct sockaddr_storage { ... long long DECLSPEC_ALIGN(8) __ss_align;
// ... } SOCKADDR_STORAGE, *PSOCKADDR_STORAGE;` is how <ws2def.h> is written.
// The class-body reader saw the decoration's parentheses, took the member for
// a method, and the struct became a class - lifted away, leaving `typedef
// SOCKADDR_STORAGE, *PSOCKADDR_STORAGE;` behind with nothing in front of the
// names. `__extension__`, which mingw writes before `__int64`, is a keyword
// that changes nothing and is now nothing here too.
#include <stdio.h>

typedef struct sockaddr_storage {
    short ss_family;
    char __ss_pad1[6];
    __extension__ long long __attribute__((aligned(8))) __ss_align;
    char __ss_pad2[112];
} SOCKADDR_STORAGE, *PSOCKADDR_STORAGE, *LPSOCKADDR_STORAGE;

typedef union _SOCKADDR_ANY {
    short family;
    SOCKADDR_STORAGE storage;
} SOCKADDR_ANY;

struct Holder {
    SOCKADDR_ANY any;
    int kind() const { return any.family; }
};

int main(void) {
    Holder h;
    h.any.storage.ss_family = 23;
    h.any.storage.__ss_align = 5;
    PSOCKADDR_STORAGE p = &h.any.storage;
    printf("%d %d %d %d\n", h.kind(), (int)p->__ss_align, (int)sizeof(SOCKADDR_STORAGE), (int)((char *)&p->__ss_align - (char *)p));
    return 0;
}
