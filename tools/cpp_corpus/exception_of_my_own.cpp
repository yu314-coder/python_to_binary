#include <stdio.h>
#include <stdexcept>
class Mine : public std::runtime_error {
public: Mine(const char *m) : std::runtime_error(m) {} };
int main(){ try { throw Mine("boom"); } catch (std::exception &e) { printf("%s\n", e.what()); } return 0; }
