/* `const Bytes frame = session->encrypted(packet);` inside
   `auto sendControlMessage = [&](const std::string &kind) { ... }` - a method
   answering an object by value, reached through a smart pointer the lambda
   captured by reference.

   A capture by reference was a pointer member, and every use of it inside
   the body was dereferenced before anything else looked at it:
   `(*this->session)->encrypted(packet)`. The pass that gives a by-value
   return the space it writes into finds its receiver by name, and that is
   not a name - so the call reached the C stage as a member call on a struct,
   refused on a line that is correct C++. The same capture spelled as a
   reference member of an ordinary class always worked, so that is what a
   capture is now: the body keeps the name, and the dereference is written
   when the method is.

   The member read after the call is the second use of the capture, which
   has to keep working too. */
#include <memory>
#include <string>
#include <vector>
#include <cstdint>
#include <cstdio>

using Bytes = std::vector<std::uint8_t>;
static const std::uint8_t kControlPacket = 7;

struct Session {
    int key = 3;
    int sendMutex = 0;
    Bytes encrypted(const Bytes &plain) {
        Bytes out;
        for (std::size_t i = 0; i < plain.size(); ++i) {
            out.push_back((std::uint8_t)(plain[i] ^ key));
        }
        return out;
    }
};

class Transport {
public:
    int serve() {
        auto session = std::make_shared<Session>();
        session->key = 5;
        auto sendControlMessage = [&](const std::string &kind) {
            const std::string encoded = kind;
            Bytes packet{kControlPacket};
            packet.insert(packet.end(), encoded.begin(), encoded.end());
            const Bytes frame = session->encrypted(packet);
            return (int)frame.size() * 100 + frame[0] + session->sendMutex;
        };
        return sendControlMessage("ab");
    }
};

int main() {
    Transport transport;
    std::printf("%d\n", transport.serve());
    return 0;
}
