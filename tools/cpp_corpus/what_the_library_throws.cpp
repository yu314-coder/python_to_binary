// What `at`, `substr`, `erase`, `insert`, `replace`, `map::at`, `stoi` and the rest throw, and what each says, as libc++ says it.
#include <cstdio>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>
#include <map>
#include <array>
#include <deque>
#include <bitset>
#include <optional>
#include <new>
#include <typeinfo>
int main() {
    std::vector<int> v(2); std::string s = "ab"; std::map<int, int> m; std::array<int, 2> a{}; std::deque<int> d; d.push_back(1);
    std::string_view sv = "ab"; std::bitset<4> b; std::optional<int> o;
    try { v.at(5); printf("vector::at: no throw\n"); }
    catch (const std::out_of_range &e) { printf("vector::at: out_of_range [%s]\n", e.what()); }
    catch (const std::invalid_argument &e) { printf("vector::at: invalid_argument [%s]\n", e.what()); }
    catch (const std::exception &e) { printf("vector::at: exception [%s]\n", e.what()); }
    try { s.at(5); printf("string::at: no throw\n"); }
    catch (const std::out_of_range &e) { printf("string::at: out_of_range [%s]\n", e.what()); }
    catch (const std::invalid_argument &e) { printf("string::at: invalid_argument [%s]\n", e.what()); }
    catch (const std::exception &e) { printf("string::at: exception [%s]\n", e.what()); }
    try { s.substr(5); printf("string::substr: no throw\n"); }
    catch (const std::out_of_range &e) { printf("string::substr: out_of_range [%s]\n", e.what()); }
    catch (const std::invalid_argument &e) { printf("string::substr: invalid_argument [%s]\n", e.what()); }
    catch (const std::exception &e) { printf("string::substr: exception [%s]\n", e.what()); }
    try { s.erase(5, 1); printf("string::erase: no throw\n"); }
    catch (const std::out_of_range &e) { printf("string::erase: out_of_range [%s]\n", e.what()); }
    catch (const std::invalid_argument &e) { printf("string::erase: invalid_argument [%s]\n", e.what()); }
    catch (const std::exception &e) { printf("string::erase: exception [%s]\n", e.what()); }
    try { s.insert(5, "x"); printf("string::insert: no throw\n"); }
    catch (const std::out_of_range &e) { printf("string::insert: out_of_range [%s]\n", e.what()); }
    catch (const std::invalid_argument &e) { printf("string::insert: invalid_argument [%s]\n", e.what()); }
    catch (const std::exception &e) { printf("string::insert: exception [%s]\n", e.what()); }
    try { s.replace(5, 1, "x"); printf("string::replace: no throw\n"); }
    catch (const std::out_of_range &e) { printf("string::replace: out_of_range [%s]\n", e.what()); }
    catch (const std::invalid_argument &e) { printf("string::replace: invalid_argument [%s]\n", e.what()); }
    catch (const std::exception &e) { printf("string::replace: exception [%s]\n", e.what()); }
    try { m.at(3); printf("map::at: no throw\n"); }
    catch (const std::out_of_range &e) { printf("map::at: out_of_range [%s]\n", e.what()); }
    catch (const std::invalid_argument &e) { printf("map::at: invalid_argument [%s]\n", e.what()); }
    catch (const std::exception &e) { printf("map::at: exception [%s]\n", e.what()); }
    try { a.at(5); printf("array::at: no throw\n"); }
    catch (const std::out_of_range &e) { printf("array::at: out_of_range [%s]\n", e.what()); }
    catch (const std::invalid_argument &e) { printf("array::at: invalid_argument [%s]\n", e.what()); }
    catch (const std::exception &e) { printf("array::at: exception [%s]\n", e.what()); }
    try { d.at(5); printf("deque::at: no throw\n"); }
    catch (const std::out_of_range &e) { printf("deque::at: out_of_range [%s]\n", e.what()); }
    catch (const std::invalid_argument &e) { printf("deque::at: invalid_argument [%s]\n", e.what()); }
    catch (const std::exception &e) { printf("deque::at: exception [%s]\n", e.what()); }
    try { std::stoi("x"); printf("stoi x: no throw\n"); }
    catch (const std::out_of_range &e) { printf("stoi x: out_of_range [%s]\n", e.what()); }
    catch (const std::invalid_argument &e) { printf("stoi x: invalid_argument [%s]\n", e.what()); }
    catch (const std::exception &e) { printf("stoi x: exception [%s]\n", e.what()); }
    try { std::stoi("99999999999"); printf("stoi big: no throw\n"); }
    catch (const std::out_of_range &e) { printf("stoi big: out_of_range [%s]\n", e.what()); }
    catch (const std::invalid_argument &e) { printf("stoi big: invalid_argument [%s]\n", e.what()); }
    catch (const std::exception &e) { printf("stoi big: exception [%s]\n", e.what()); }
    try { std::stol("x"); printf("stol x: no throw\n"); }
    catch (const std::out_of_range &e) { printf("stol x: out_of_range [%s]\n", e.what()); }
    catch (const std::invalid_argument &e) { printf("stol x: invalid_argument [%s]\n", e.what()); }
    catch (const std::exception &e) { printf("stol x: exception [%s]\n", e.what()); }
    try { std::stoul("x"); printf("stoul x: no throw\n"); }
    catch (const std::out_of_range &e) { printf("stoul x: out_of_range [%s]\n", e.what()); }
    catch (const std::invalid_argument &e) { printf("stoul x: invalid_argument [%s]\n", e.what()); }
    catch (const std::exception &e) { printf("stoul x: exception [%s]\n", e.what()); }
    try { std::stoll("999999999999999999999"); printf("stoll big: no throw\n"); }
    catch (const std::out_of_range &e) { printf("stoll big: out_of_range [%s]\n", e.what()); }
    catch (const std::invalid_argument &e) { printf("stoll big: invalid_argument [%s]\n", e.what()); }
    catch (const std::exception &e) { printf("stoll big: exception [%s]\n", e.what()); }
    try { std::stod("x"); printf("stod x: no throw\n"); }
    catch (const std::out_of_range &e) { printf("stod x: out_of_range [%s]\n", e.what()); }
    catch (const std::invalid_argument &e) { printf("stod x: invalid_argument [%s]\n", e.what()); }
    catch (const std::exception &e) { printf("stod x: exception [%s]\n", e.what()); }
    try { std::stof("x"); printf("stof x: no throw\n"); }
    catch (const std::out_of_range &e) { printf("stof x: out_of_range [%s]\n", e.what()); }
    catch (const std::invalid_argument &e) { printf("stof x: invalid_argument [%s]\n", e.what()); }
    catch (const std::exception &e) { printf("stof x: exception [%s]\n", e.what()); }
    try { printf("%d\n", std::stoi("-12abc")); printf("stoi neg: no throw\n"); }
    catch (const std::out_of_range &e) { printf("stoi neg: out_of_range [%s]\n", e.what()); }
    catch (const std::invalid_argument &e) { printf("stoi neg: invalid_argument [%s]\n", e.what()); }
    catch (const std::exception &e) { printf("stoi neg: exception [%s]\n", e.what()); }
    try { printf("%lu\n", std::stoul("-1")); printf("stoul neg: no throw\n"); }
    catch (const std::out_of_range &e) { printf("stoul neg: out_of_range [%s]\n", e.what()); }
    catch (const std::invalid_argument &e) { printf("stoul neg: invalid_argument [%s]\n", e.what()); }
    catch (const std::exception &e) { printf("stoul neg: exception [%s]\n", e.what()); }
    try { printf("%d\n", std::stoi("-2147483648")); printf("stoi min: no throw\n"); }
    catch (const std::out_of_range &e) { printf("stoi min: out_of_range [%s]\n", e.what()); }
    catch (const std::invalid_argument &e) { printf("stoi min: invalid_argument [%s]\n", e.what()); }
    catch (const std::exception &e) { printf("stoi min: exception [%s]\n", e.what()); }
    try { printf("%d\n", std::stoi("2147483648")); printf("stoi over: no throw\n"); }
    catch (const std::out_of_range &e) { printf("stoi over: out_of_range [%s]\n", e.what()); }
    catch (const std::invalid_argument &e) { printf("stoi over: invalid_argument [%s]\n", e.what()); }
    catch (const std::exception &e) { printf("stoi over: exception [%s]\n", e.what()); }
    try { throw std::exception(); printf("exception: no throw\n"); }
    catch (const std::out_of_range &e) { printf("exception: out_of_range [%s]\n", e.what()); }
    catch (const std::invalid_argument &e) { printf("exception: invalid_argument [%s]\n", e.what()); }
    catch (const std::exception &e) { printf("exception: exception [%s]\n", e.what()); }
    try { throw std::bad_alloc(); printf("bad_alloc: no throw\n"); }
    catch (const std::out_of_range &e) { printf("bad_alloc: out_of_range [%s]\n", e.what()); }
    catch (const std::invalid_argument &e) { printf("bad_alloc: invalid_argument [%s]\n", e.what()); }
    catch (const std::exception &e) { printf("bad_alloc: exception [%s]\n", e.what()); }
    try { throw std::bad_cast(); printf("bad_cast: no throw\n"); }
    catch (const std::out_of_range &e) { printf("bad_cast: out_of_range [%s]\n", e.what()); }
    catch (const std::invalid_argument &e) { printf("bad_cast: invalid_argument [%s]\n", e.what()); }
    catch (const std::exception &e) { printf("bad_cast: exception [%s]\n", e.what()); }
    try { throw std::runtime_error(std::string("rt")); printf("runtime: no throw\n"); }
    catch (const std::out_of_range &e) { printf("runtime: out_of_range [%s]\n", e.what()); }
    catch (const std::invalid_argument &e) { printf("runtime: invalid_argument [%s]\n", e.what()); }
    catch (const std::exception &e) { printf("runtime: exception [%s]\n", e.what()); }
    try { throw std::logic_error("lg"); printf("logic: no throw\n"); }
    catch (const std::out_of_range &e) { printf("logic: out_of_range [%s]\n", e.what()); }
    catch (const std::invalid_argument &e) { printf("logic: invalid_argument [%s]\n", e.what()); }
    catch (const std::exception &e) { printf("logic: exception [%s]\n", e.what()); }
    try { sv.at(5); printf("sv at: no throw\n"); }
    catch (const std::out_of_range &e) { printf("sv at: out_of_range [%s]\n", e.what()); }
    catch (const std::invalid_argument &e) { printf("sv at: invalid_argument [%s]\n", e.what()); }
    catch (const std::exception &e) { printf("sv at: exception [%s]\n", e.what()); }
    try { sv.substr(5); printf("sv substr: no throw\n"); }
    catch (const std::out_of_range &e) { printf("sv substr: out_of_range [%s]\n", e.what()); }
    catch (const std::invalid_argument &e) { printf("sv substr: invalid_argument [%s]\n", e.what()); }
    catch (const std::exception &e) { printf("sv substr: exception [%s]\n", e.what()); }
    try { b.test(9); printf("bitset test: no throw\n"); }
    catch (const std::out_of_range &e) { printf("bitset test: out_of_range [%s]\n", e.what()); }
    catch (const std::invalid_argument &e) { printf("bitset test: invalid_argument [%s]\n", e.what()); }
    catch (const std::exception &e) { printf("bitset test: exception [%s]\n", e.what()); }
    try { b.set(9); printf("bitset set: no throw\n"); }
    catch (const std::out_of_range &e) { printf("bitset set: out_of_range [%s]\n", e.what()); }
    catch (const std::invalid_argument &e) { printf("bitset set: invalid_argument [%s]\n", e.what()); }
    catch (const std::exception &e) { printf("bitset set: exception [%s]\n", e.what()); }
    try { o.value(); printf("optional: no throw\n"); }
    catch (const std::out_of_range &e) { printf("optional: out_of_range [%s]\n", e.what()); }
    catch (const std::invalid_argument &e) { printf("optional: invalid_argument [%s]\n", e.what()); }
    catch (const std::exception &e) { printf("optional: exception [%s]\n", e.what()); }
    try { throw std::out_of_range("x"); } catch (const std::logic_error &e) { printf("logic caught %s\n", e.what()); }
    std::runtime_error r("one"); std::runtime_error r2 = r; r = std::runtime_error("two"); printf("%s %s\n", r.what(), r2.what());
    return 0;
}
