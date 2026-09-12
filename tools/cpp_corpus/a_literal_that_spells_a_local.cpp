// A string literal that spells the name of a local declared further down.
// Where an object is declared was read off the raw text, so `"proof"` as a
// key in a call counted as the declaration of `proof` - and the early
// return above it then took apart an object that did not exist yet, which
// the C stage reported as a name declared nowhere on a line that is
// correct C++.
#include <vector>
#include <string>
#include <cstdio>

static std::string pick(const std::string &json, const char *key) {
    return json + key;
}

static void handle(bool bad) {
    std::vector<int> held;
    held.push_back(1);
    // The literal spells the name of a local declared further down.
    const std::string kind = pick("k", "proof");
    if (bad) { std::printf("refused\n"); return; }
    const std::vector<int> proof = held;
    std::printf("%d %s\n", (int)proof.size(), kind.c_str());
}

int main() { handle(true); handle(false); return 0; }
