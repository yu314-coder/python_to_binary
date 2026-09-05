// The text inside a frame read off a socket: a vector of bytes, and a string
// built from the range past its first byte. The range constructor took
// `char *` exactly, and nothing that arrives over a wire is spelled so.
#include <cstdio>
#include <cstdint>
#include <string>
#include <vector>

int main() {
    std::vector<uint8_t> frame;
    frame.push_back(7);
    const char *text = "{\"kind\":\"hello\"}";
    for (int i = 0; text[i] != 0; i++) frame.push_back((uint8_t)text[i]);
    std::string body(frame.begin() + 1, frame.end());
    std::string whole(frame.begin(), frame.end());
    printf("%s %d %d\n", body.c_str(), (int)body.size(), (int)whole.size());
    return 0;
}
