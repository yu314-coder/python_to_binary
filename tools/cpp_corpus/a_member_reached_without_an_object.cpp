// A method of the class called by its bare name, and its result written to a
// stream. C++ resolves the name through `this`; py2bin left it a bare call,
// so the pass that gives a value-returning member somewhere to put its
// answer never saw it and the C compiler was handed a call with one argument
// to a function taking two. The overload of `<<` is chosen from what the
// class says the member answers, which is the only place it is written.
#include <cstdio>
#include <string>
#include <sstream>

class Report {
public:
    std::string code_;
    int count_;
    Report() { code_ = "1234"; count_ = 7; }
    std::string code() const;
    int count() const { return count_; }
    std::string line() const {
        std::ostringstream out;
        out << "code=" << code() << " count=" << count() << ";";
        return out.str();
    }
};

std::string Report::code() const { return code_; }

int main() {
    Report r;
    printf("%s\n", r.line().c_str());
    return 0;
}
