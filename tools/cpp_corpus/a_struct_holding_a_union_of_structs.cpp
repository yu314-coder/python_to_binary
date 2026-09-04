// A plain struct holding a union that holds plain structs, in the order written.
//
// <winsock2.h> writes `typedef union sockaddr_gen { struct sockaddr Address;
// struct sockaddr_in AddressIn; } sockaddr_gen;` and then `typedef struct
// _INTERFACE_INFO { u_long iiFlags; sockaddr_gen iiAddress; } INTERFACE_INFO;`.
// The translator hoisted enums and unions as one group and plain structs as
// another, so whichever group went first, something in it named a type from
// the other. They go out in the order they were written now, which is valid
// C whenever the C++ was.
#include <stdio.h>

typedef enum { AF_NONE = 0, AF_INET = 2 } family_t;

typedef struct sockaddr { unsigned short sa_family; char sa_data[14]; } sockaddr;
typedef struct sockaddr_in { unsigned short sin_family; unsigned short sin_port; unsigned int sin_addr; } sockaddr_in;

typedef union sockaddr_gen {
    sockaddr Address;
    sockaddr_in AddressIn;
} sockaddr_gen;

typedef struct _INTERFACE_INFO {
    unsigned long iiFlags;
    sockaddr_gen iiAddress;
    sockaddr_gen iiBroadcastAddress;
    family_t family;
} INTERFACE_INFO, *LPINTERFACE_INFO;

struct Table {
    INTERFACE_INFO rows[2];
    int count() const { return 2; }
};

int main(void) {
    Table t;
    t.rows[0].iiAddress.AddressIn.sin_port = 80;
    t.rows[1].iiBroadcastAddress.Address.sa_family = AF_INET;
    t.rows[1].family = AF_INET;
    LPINTERFACE_INFO p = &t.rows[1];
    printf("%d %d %d %d %d\n", t.count(), t.rows[0].iiAddress.AddressIn.sin_port, p->iiBroadcastAddress.Address.sa_family, (int)p->family, (int)sizeof(INTERFACE_INFO));
    return 0;
}
