#include <cstdio>
struct Token { long long v; };
class Session {
public:
    int count_ = 3;
    bool ready_ = true;
    Token token_{};
    const char *name_ = "sess";
    int Bump() { count_ += 1; return count_; }
};
int main() {
    Session s;
    printf("%d %d %lld %s\n", s.Bump(), (int)s.ready_, s.token_.v, s.name_);
    return 0;
}
