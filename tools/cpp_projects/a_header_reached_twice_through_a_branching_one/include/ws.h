/* A guarded header that branches on __cplusplus, the way an SDK's does. */
#ifndef WS_H
#define WS_H
#ifdef __cplusplus
extern "C" {
#else
typedef int ws_plain_c_marker;
#endif
typedef enum { COMP_EQUAL = 0, COMP_NOTLESS } WSAECOMPARATOR;
static inline int compare_kind(int x) { return x ? COMP_NOTLESS : COMP_EQUAL; }
#ifdef __cplusplus
}
#else
static ws_plain_c_marker ws_marker;
#endif
#endif
