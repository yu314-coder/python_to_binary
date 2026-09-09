// `char text[64]; ... localAddress_ = text;` - a buffer filled by a call and
// then kept in a string.
//
// py2bin reads one flat text: the body, the scope around it, and every header
// it pasted. Asked what `text` is with no position to be nearest to, it
// answered with the *last* declaration of that name anywhere - which is
// `string text;`, a member of the `path` class inside py2bin's own
// <filesystem>. So the value was said to be a string already: nothing was
// converted, nothing was constructed, and the C stage was handed a `char *`
// where a struct goes.
//
// The body's own declarations are the nearer ones, and are asked first.
#include <cstdio>
#include <filesystem>
#include <string>

class Host {
public:
    std::string localAddress_;
    std::string name_;

    void look() {
        char text[64];
        text[0] = 'a'; text[1] = 'b'; text[2] = 0;
        localAddress_ = text;
    }

    // The same name again, and a real path beside it, so the header's class
    // is in the unit and its member is there to be found.
    void named() {
        const char *text = "given";
        name_ = text;
    }
};

int main() {
    Host host;
    host.look();
    host.named();
    std::filesystem::path where("/tmp/one");
    printf("%s %d|%s|%s\n", host.localAddress_.c_str(),
           (int)host.localAddress_.size(), host.name_.c_str(),
           where.filename().string().c_str());
    return 0;
}
