/* Windows spells a socket `SOCKET`, which is `UINT_PTR` - an unsigned
   integer wide enough to hold a pointer, and never followed. The vetted
   table called every one of these a pointer, so a program that declared them
   the way Windows declares them was refused for agreeing with Windows. They
   are handles now: a pointer or a word of the same width, and a plain `int`
   still refused. The address a call is given is a real pointer, and stays
   one. */
#include <stdio.h>

#ifdef _WIN32
typedef unsigned long long SOCKET;

struct sockaddr { unsigned short family; char data[14]; };

SOCKET socket(int family, int type, int protocol);
int bind(SOCKET s, const struct sockaddr *name, int length);
int listen(SOCKET s, int backlog);
int shutdown(SOCKET s, int how);
int closesocket(SOCKET s);

static int listen_on(void) {
    struct sockaddr address;
    SOCKET listener = socket(2, 1, 6);
    int index;
    if (listener == (SOCKET)(~0)) {
        return 1;
    }
    address.family = 2;
    for (index = 0; index < 14; index++) {
        address.data[index] = 0;
    }
    if (bind(listener, &address, (int)sizeof(address)) != 0) {
        closesocket(listener);
        return 2;
    }
    if (listen(listener, 1) != 0) {
        closesocket(listener);
        return 3;
    }
    shutdown(listener, 2);
    closesocket(listener);
    return 4;
}
#else
static int listen_on(void) {
    return 4;
}
#endif

int main(void) {
    printf("%d\n", listen_on() > 0);
    return 0;
}
