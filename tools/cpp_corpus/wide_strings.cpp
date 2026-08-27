/* Windows is wide throughout - every `W` entry point takes a wchar_t string -
   so a program that talks to it holds its strings that way. `L"x"` is one
   operand and not a name called L followed by a string, which is how it was
   read: blanked for scanning it left an `L` standing, and an `L` reads as a
   name. A declaration was taken for a copy of a variable called L, and a
   call for a call on one. */
#include <stdio.h>
#include <string>

int main() {
    std::wstring filled(4, L'z');
    std::wstring greeting = L"hello";
    greeting += L", world";
    greeting += L'!';
    std::wstring joined = greeting + L" again";
    filled.resize(2);

    std::string narrow = "abc";
    narrow += "de";

    printf("%d %d %d %d %d %d\n", (int)filled.size(), (int)greeting.size(),
           (int)joined.size(), (int)greeting.c_str()[0], (int)narrow.size(),
           (int)filled.c_str()[0]);
    return 0;
}
