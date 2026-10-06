// A handler takes the type it names: `throw 'c'` is caught by `catch (char)`, not by the `catch (int)` written first.
#include <cstdio>
int main() {
    int got = 0;
    try { throw 42; } catch (int x) { got = x; }
    try { throw 'c'; } catch (int x) { got += 1000; } catch (char c) { got += c; }
    printf("%d\n", got);
    return 0;
}
