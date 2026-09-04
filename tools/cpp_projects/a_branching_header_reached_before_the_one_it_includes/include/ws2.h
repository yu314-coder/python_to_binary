/* Another branching header that includes the first, as ws2tcpip.h includes winsock2.h. */
#ifndef WS2_H
#define WS2_H
#include <ws.h>
#ifdef __cplusplus
extern "C" {
#else
typedef int ws2_plain_c_marker;
#endif
static inline int address_kind(int x) { return compare_kind(x) + 10; }
#ifdef __cplusplus
}
#else
static ws2_plain_c_marker ws2_marker;
#endif
#endif
