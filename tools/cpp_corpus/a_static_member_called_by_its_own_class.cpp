/* `payload << escapeJson(state)` inside a member of a class whose
   `escapeJson` is `static`. A bare call to one of the class's own members
   means `this->` - except a static one, which is never given the object. The
   pass that gives a member call its receiver back, so that an answer used as
   an object is written through space the caller provides, named this one
   too: it reached the C as a call handed `this` as well as that space, one
   argument more than the function takes.

   The stream is one place its answer is used as an object; the initialiser
   below is the other, and a static returning a plain int is the third case,
   which always worked and has to keep working. */
#include <sstream>
#include <string>
#include <cstdio>

class Session {
public:
    void sendInitialState(const char *state, const std::string &headline);
    int calls = 0;
private:
    static std::string escapeJson(const std::string &value);
    static int twice(int v) { return v * 2; }
};

std::string Session::escapeJson(const std::string &value) {
    std::string result;
    for (std::size_t i = 0; i < value.size(); ++i) {
        if (value[i] == '"') result.push_back('\\');
        result.push_back(value[i]);
    }
    return result;
}

void Session::sendInitialState(const char *state, const std::string &headline) {
    std::ostringstream payload;
    payload << "{\"state\":\"" << escapeJson(state) << "\""
            << ",\"headline\":\"" << escapeJson(headline) << "\"}";
    const std::string plain = escapeJson("a\"b");
    calls = twice(calls + 1);
    std::printf("%s %s %d\n", payload.str().c_str(), plain.c_str(), calls);
}

int main() {
    Session session;
    session.sendInitialState("idle", "Waiting \"now\"");
    return 0;
}
