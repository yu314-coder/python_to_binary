#include <cstdio>
#include <cstddef>
#pragma pack(push, 1)
struct Wire { char tag; int len; short crc; int area() const { return len; } };
extern "C" { struct Hdr { char a; int b; char c; }; }
#pragma pack(pop)
struct After { char a; int b; };
int main() {
    Wire w; w.tag = 1; w.len = 9; w.crc = 2;
    printf("%d %d %d %d %d %d\n", (int)sizeof(Wire), (int)offsetof(Wire, len),
           (int)sizeof(Hdr), (int)offsetof(Hdr, b), (int)offsetof(Hdr, c), (int)sizeof(After));
    return 0;
}
