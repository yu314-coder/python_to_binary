// <fstream>: a file written through `<<` and read back with `>>` and getline.
//
// py2bin refused <fstream> by name; it is shipped now, straight over the C
// file rather than through py2bin's 255-character string - a file read into
// one of those would have been cut without a word. Written with the mode
// constants a program spells as std::ios::app, read as words, numbers and
// lines, tested for openness and end, and the file removed afterwards through
// <filesystem> so the corpus directory is left as it was found.
#include <fstream>
#include <filesystem>
#include <string>
#include <cstdio>

int main() {
    {
        std::ofstream out("a_stream_probe.txt");
        out << "hello " << 42 << " " << 2.5 << "\n";
        out << "second line here\n";
        out.close();
        std::ofstream more("a_stream_probe.txt", std::ios::app);
        more << "third\n";
    }
    std::ifstream in("a_stream_probe.txt");
    std::string word;
    int n = 0;
    double d = 0;
    in >> word >> n >> d;
    std::string line;
    std::getline(in, line);
    std::getline(in, line);
    std::string last;
    int lines = 0;
    while (std::getline(in, last)) lines += 1;
    printf("%s %d %.1f [%s] %s %d %d %d\n", word.c_str(), n, d, line.c_str(), last.c_str(), lines, (int)in.is_open(), (int)in.eof());
    in.close();
    std::ifstream missing("a_stream_probe_missing.txt");
    printf("%d %d\n", (int)missing.is_open(), (int)!missing);
    std::filesystem::remove("a_stream_probe.txt");
    return 0;
}
