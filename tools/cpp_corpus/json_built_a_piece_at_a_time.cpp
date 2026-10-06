// Twenty-nine thousand characters of JSON built a piece at a time: the shape of what a program sends over a socket.
#include <cstdio>
#include <string>
int main() {
    std::string json = "{";
    for (int i = 0; i < 2000; i++) { if (i) json += ","; json += "\"k" + std::to_string(i) + "\":" + std::to_string(i * i); }
    json += "}";
    size_t sum = 0; for (char ch : json) sum += (unsigned char)ch;
    printf("%zu %zu %c %s\n", json.size(), sum, json[json.size() - 2], json.substr(json.size() - 14).c_str());
    return 0;
}
