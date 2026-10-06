// A copied string is a string of its own - declared, assigned, handed to a function, held in a struct or a vector - and changing the copy leaves the original alone.
#include <cstdio>
#include <string>
#include <vector>
std::string shout(std::string s) { s += "!"; return s; }
struct Named { std::string name; int n; };
int main() {
    std::string a = "hello";
    std::string b = a; b[0] = 'J';
    std::string c; c = a; c += " world";
    std::string d = shout(a);
    Named x{"first", 1}; Named y = x; y.name += "-copy";
    std::vector<std::string> v; v.push_back(a); v.push_back(c); std::vector<std::string> w = v; w[0][0] = 'Y';
    printf("%s|%s|%s|%s|%s|%s|%s|%s\n", a.c_str(), b.c_str(), c.c_str(), d.c_str(), x.name.c_str(), y.name.c_str(), v[0].c_str(), w[0].c_str());
    return 0;
}
