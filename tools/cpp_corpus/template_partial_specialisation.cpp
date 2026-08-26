/* A class template written again for a shape of argument rather than for
   every argument. Every entry in a traits header is one of these, and which
   copy a use gets is decided by which pattern is narrower. */
#include <stdio.h>

template <class T> struct Name { static const int id = 0; };
template <> struct Name<int> { static const int id = 1; };
template <class T> struct Name<T *> { static const int id = 2; };
template <class T> struct Name<T **> { static const int id = 3; };
template <class T> struct Name<const T> { static const int id = 4; };

template <class T, class U> struct Pair { static const int kind = 0; };
template <class T> struct Pair<T, T> { static const int kind = 1; };
template <class U> struct Pair<int, U> { static const int kind = 2; };
template <> struct Pair<int, int> { static const int kind = 3; };

template <class T> struct strip { typedef T type; };
template <class T> struct strip<T &> { typedef T type; };
template <class T> struct strip<T *> { typedef T type; };

int main() {
    printf("%d %d %d %d %d\n", Name<double>::id, Name<int>::id,
           Name<double *>::id, Name<double **>::id, Name<const double>::id);
    printf("%d %d %d %d\n", Pair<double, char>::kind, Pair<char, char>::kind,
           Pair<int, char>::kind, Pair<int, int>::kind);
    strip<int &>::type a = 7;
    strip<char *>::type b = 'q';
    strip<double>::type c = 1.5;
    printf("%d %c %.1f\n", a, b, c);
    return 0;
}
