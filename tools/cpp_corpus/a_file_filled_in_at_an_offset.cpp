// `std::fstream f(p, ios::binary | ios::in | ios::out)` - a file open for
// both at once, which is how a program fills one in at an offset as the
// pieces of it arrive. Neither of the two py2bin shipped can do that: one
// only reads, and the other empties the file to write it. That pair of flags
// keeps what is there and does not create the file, which is what C++ says of
// them, so the open underneath has a third way of being asked.
//
// One position, as the file has: py2bin keeps no separate read and write
// cursor, so `seekg` and `seekp` are the same move. A program that fills a
// file in at offsets, which is what this is for, cannot tell.
#include <cstdio>
#include <fstream>
#include <filesystem>
#include <string>

int main() {
    const char *where = "a_filled_in_probe.bin";
    {
        std::ofstream made(where, std::ios::binary | std::ios::trunc);
        made.write("abcdefgh", 8);
    }

    {
        std::fstream file(where, std::ios::binary | std::ios::in | std::ios::out);
        if (!file) { printf("not open\n"); return 1; }
        // The offsets a program writes with the names it writes them under.
        file.seekp(static_cast<std::streamoff>(3));
        file.write("XY", static_cast<std::streamsize>(2));
        file.seekp(static_cast<std::streamoff>(0), std::ios::end);
        file.write("!", 1);
    }

    std::fstream back(where, std::ios::binary | std::ios::in | std::ios::out);
    char buffer[32];
    back.seekg(0);
    back.read(buffer, 9);
    buffer[back.gcount()] = 0;
    long read_back = back.gcount();
    long long at = back.tellg();
    int open_now = back.is_open() ? 1 : 0;
    int well = back.good() ? 1 : 0;
    back.close();

    // A file that is not there is not opened at all by that pair of flags.
    std::fstream missing("a_file_that_is_not_here.bin",
                         std::ios::binary | std::ios::in | std::ios::out);
    int refused = !missing;

    printf("%s %d %d %d %d %d\n", buffer, (int)read_back, (int)at, open_now,
           well, refused);
    std::filesystem::remove(where);
    return 0;
}
