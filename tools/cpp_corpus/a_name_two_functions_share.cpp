// Two functions, each with a local called `output`, and they are not the
// same type. The reader that types a name reads the whole unit and, with no
// position to be nearest to, takes the LAST declaration anywhere - so the
// `DATA_BLOB output` here lost to the `std::string output` below it, the
// member read off it had no type, and the constructor the call meant could
// not be chosen.
#include <vector>
#include <string>
#include <cstdint>
#include <cstdio>

typedef struct _CRYPTOAPI_BLOB {
    unsigned long cbData;
    std::uint8_t *pbData;
} CRYPT_INTEGER_BLOB, *PCRYPT_INTEGER_BLOB, DATA_BLOB, *PDATA_BLOB;

using Bytes = std::vector<std::uint8_t>;

static Bytes load() {
    Bytes held;
    held.push_back(1); held.push_back(2); held.push_back(3);
    DATA_BLOB output{};
    output.pbData = held.data();
    output.cbData = (unsigned long)held.size();
    Bytes result(output.pbData, output.pbData + output.cbData);
    return result;
}

// A later function whose `output` is something else entirely.
static void say() {
    std::string output = "state";
    std::printf("%s\n", output.c_str());
}

int main() { std::printf("%d\n", (int)load().size()); say(); return 0; }
