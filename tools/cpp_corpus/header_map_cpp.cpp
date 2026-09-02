/* <map>: keyed by a string, walked in key order, and asked whether a key is
   there. py2bin's own keeps insertion order for `unordered_map` and key
   order here, which is what C++ promises of this one. */
#include <map>
#include <string>
#include <cstdio>

int main() {
    std::map<std::string, int> held;
    held["b"] = 2;
    held["a"] = 1;
    held["c"] = 3;
    printf("%d %d\n", (int)held.size(), held["a"]);
    for (std::map<std::string, int>::iterator it = held.begin();
         it != held.end(); ++it) {
        printf("%s=%d ", it->first.c_str(), it->second);
    }
    printf("\n");
    printf("%d %d\n", (int)held.count("b"), (int)held.count("z"));
    return 0;
}
