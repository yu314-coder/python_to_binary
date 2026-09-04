/* A member given an alignment, and a struct given one, laid out as the compiler lays them.
 *
 * `long long __attribute__((aligned(8))) __ss_align;` is how <ws2def.h>
 * writes sockaddr_storage (through DECLSPEC_ALIGN), and py2bin refused the
 * attribute by name rather than lay the struct out wrongly. It is laid out
 * now: a member starts at the next multiple of the larger of its own
 * alignment and the one asked for, and the whole struct is padded to the
 * strictest of them. Offsets are printed as pointer differences, since
 * offsetof is not a constant here. */
#include <stdio.h>
#define OFF(s, m) ((int)((char *)&(s).m - (char *)&(s)))
struct Mid { short f; char pad[6]; long long __attribute__((aligned(8))) a; char tail[3]; };
struct Front { char c; __attribute__((aligned(16))) int v; char d; };
struct Wide { char c; long long __attribute__((aligned(32))) big; };
struct __attribute__((aligned(16))) Whole { char c; short s; };
union U { char c; long long __attribute__((aligned(16))) x; };
struct Nest { char c; struct Whole w; char d; };
int main(void) {
    struct Mid m; struct Front f; struct Wide w; struct Whole h; struct Nest n; struct Whole arr[3];
    printf("%d %d %d %d\n", (int)sizeof(struct Mid), OFF(m, pad), OFF(m, a), OFF(m, tail));
    printf("%d %d %d\n", (int)sizeof(struct Front), OFF(f, v), OFF(f, d));
    printf("%d %d\n", (int)sizeof(struct Wide), OFF(w, big));
    printf("%d %d %d\n", (int)sizeof(struct Whole), OFF(h, s), (int)sizeof(union U));
    printf("%d %d %d %d\n", (int)sizeof(struct Nest), OFF(n, w), OFF(n, d), (int)((char *)&arr[1] - (char *)&arr[0]));
    return 0;
}
