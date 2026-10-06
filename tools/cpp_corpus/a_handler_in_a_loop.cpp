// A try inside a loop body: reaching the handler leaves the vector declared before the loop alone.
#include <cstdio>
#include <vector>
#include <string>
struct E { int c; E(int v) : c(v) {} };
int main() {
    std::vector<std::string> log;
    for (int i = 0; i < 5; i++) {
        try { std::string s = "item" + std::to_string(i); if (i == 2) throw E(i); log.push_back(s); }
        catch (const E &e) { log.push_back("error" + std::to_string(e.c)); }
    }
    for (auto &s : log) printf("%s ", s.c_str());
    printf("\n");
    return 0;
}
