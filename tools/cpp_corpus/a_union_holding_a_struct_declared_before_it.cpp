// A union whose member is a struct typedef'd above it, the way an SDK writes one.
//
// `typedef struct sockaddr_in { ... } SOCKADDR_IN;` and then `typedef union
// _SOCKADDR_INET { SOCKADDR_IN Ipv4; ... } SOCKADDR_INET;` is how <ws2ipdef.h>
// is written. The translator hoists every enum and union above the plain
// structs, so the union was emitted before the struct it holds and the C
// compiler met a type it had not seen. A tagged type that names a plain
// struct's type now goes after the plain structs. A class holding the union
// afterwards, and an enum the struct holds, keep their places.
#include <stdio.h>

typedef enum _ADDRESS_KIND { KIND_NONE = 0, KIND_V4 = 2, KIND_V6 = 23 } ADDRESS_KIND;

typedef struct sockaddr_in {
    ADDRESS_KIND sin_family;
    unsigned short sin_port;
    unsigned int sin_addr;
} SOCKADDR_IN, *PSOCKADDR_IN;

typedef struct sockaddr_in6 {
    ADDRESS_KIND sin6_family;
    unsigned short sin6_port;
    unsigned char sin6_addr[16];
} SOCKADDR_IN6;

typedef union _SOCKADDR_INET {
    SOCKADDR_IN Ipv4;
    SOCKADDR_IN6 Ipv6;
    ADDRESS_KIND si_family;
} SOCKADDR_INET, *PSOCKADDR_INET;

struct Endpoint {
    SOCKADDR_INET address;
    int port() const { return address.si_family == KIND_V4 ? address.Ipv4.sin_port : address.Ipv6.sin6_port; }
};

int main(void) {
    Endpoint e;
    e.address.Ipv4.sin_family = KIND_V4;
    e.address.Ipv4.sin_port = 8080;
    e.address.Ipv4.sin_addr = 7;
    PSOCKADDR_INET p = &e.address;
    printf("%d %d %d %d\n", e.port(), (int)p->si_family, (int)sizeof(SOCKADDR_INET) >= (int)sizeof(SOCKADDR_IN6), (int)p->Ipv4.sin_addr);
    return 0;
}
