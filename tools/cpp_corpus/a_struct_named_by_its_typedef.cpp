// `typedef struct _TAG { ... } A, *PA, B, *PB;` - one body and several
// names, which is how C-style C++ and every Windows header writes a struct.
// The reader looks a class body up by the name in front of its brace, so
// `output.pbData` on a `DATA_BLOB` had no type at all - and the call it
// stood in could not be read, so a template's copy for it was never made.
#include <vector>
#include <cstdint>
#include <cstdio>

typedef struct _CRYPTOAPI_BLOB {
    unsigned long cbData;
    std::uint8_t *pbData;
} CRYPT_INTEGER_BLOB, *PCRYPT_INTEGER_BLOB,
  DATA_BLOB,          *PDATA_BLOB;

using Bytes = std::vector<std::uint8_t>;

int main() {
    Bytes held;
    for (int index = 0; index < 4; index++) {
        held.push_back((std::uint8_t)(index + 7));
    }

    DATA_BLOB output{};
    output.pbData = held.data();
    output.cbData = (unsigned long)held.size();

    Bytes result(output.pbData, output.pbData + output.cbData);
    std::printf("%d %d %d\n", (int)result.size(), (int)result[0], (int)result[3]);

    // The first name in the list reads the same body.
    CRYPT_INTEGER_BLOB other{};
    other.pbData = held.data();
    other.cbData = 2;
    Bytes shorter(other.pbData, other.pbData + other.cbData);
    std::printf("%d %d\n", (int)shorter.size(), (int)shorter[1]);
    return 0;
}
