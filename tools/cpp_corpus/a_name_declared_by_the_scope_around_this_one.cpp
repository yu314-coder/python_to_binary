// A block is translated on its own, so a name the function around it
// declared is not in its text. Read from the whole unit instead, `text` was
// answered with a member of that name inside a shipped header - and the
// assignment below was left handing a `char *` where a string goes.
#include <cstdio>
#include <cstring>
#include <string>
#include <filesystem>

class Holder {
public:
    std::string localAddress_;
    std::string deepest_;
    void find();
    void deep();
    void shadowed();
};

void Holder::find() {
    for (int i = 0; i < 3; ++i) {
        char text[16]{};
        std::snprintf(text, sizeof(text), "10.0.0.%d", i);
        if (std::strncmp(text, "127.", 4) != 0) { localAddress_ = text; break; }
    }
}

// Two blocks out, not one.
void Holder::deep() {
    char text[16] = "203.0.113.7";
    {
        {
            if (text[0] == '2') { deepest_ = text; }
        }
    }
}

// And the nearest declaration is still the one in view.
void Holder::shadowed() {
    char text[16] = "203.0.113.7";
    {
        const char *text = "198.51.100.4";
        localAddress_ = text;
    }
    std::printf("%s\n", text);
}

int main() {
    Holder h;
    h.find();
    std::printf("%s\n", h.localAddress_.c_str());
    h.deep();
    std::printf("%s\n", h.deepest_.c_str());
    h.shadowed();
    std::printf("%s\n", h.localAddress_.c_str());
    return 0;
}
