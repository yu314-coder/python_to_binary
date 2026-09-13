/* `macID_(computerID())` - a member built from what a free function answers
   by value, in a constructor whose initialiser list builds members on either
   side of it.

   The body is rewritten with the list still in it, so the temporary that
   call needs was written in front of that entry, where it belongs. Then the
   list was taken out and every member built at the top of the body - and the
   temporary was left behind, below them: `this->macID_ = *&__py2bin_value_1;`
   came before `__py2bin_value_1` was declared, and the C stage said so. */
#include <cstdio>
#include <functional>
#include <string>

static std::string computerID() {
    return std::string("host-") + "42";
}

class Transport {
public:
    explicit Transport(std::function<void(int)> handler)
        : handler_(std::move(handler)), pairingCode_(), macID_(computerID()) {}
    std::string describe() const {
        return pairingCode_ + "|" + macID_ + "|" + localAddress_;
    }
private:
    std::function<void(int)> handler_;
    std::string pairingCode_;
    std::string macID_;
    std::string localAddress_;
};

int main() {
    Transport transport([](int) {});
    std::printf("%s\n", transport.describe().c_str());
    return 0;
}
