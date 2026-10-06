// A string edited in place: replace, insert, erase, append, substr, resize, pop_back and clear, each as libc++ does it.
#include <cstdio>
#include <string>
int main() {
    std::string s = "hello world";
    s.replace(0, 5, "HELLO"); printf("[%s]\n", s.c_str());
    s.insert(5, ","); printf("[%s]\n", s.c_str());
    s.insert(0, 3, '>'); printf("[%s]\n", s.c_str());
    s.erase(0, 3); printf("[%s]\n", s.c_str());
    s.erase(5); printf("[%s]\n", s.c_str());
    s.append(" there").append(2, '!'); printf("[%s]\n", s.c_str());
    std::string t = s.substr(6, 5); printf("[%s] [%s]\n", t.c_str(), s.substr(6).c_str());
    s.resize(3); printf("[%s] %zu\n", s.c_str(), s.size());
    s.resize(6, '.'); printf("[%s] %zu\n", s.c_str(), s.size());
    s.pop_back(); s.push_back('?'); printf("[%s] %c %c\n", s.c_str(), s.front(), s.back());
    s.clear(); printf("[%s] %d\n", s.c_str(), (int)s.empty());
    return 0;
}
