/* `encoder->Initialize(stream, WICBitmapEncoderNoCache)` - an `IWICStream *`
   handed to a parameter that is an `IStream *`, through an interface's table.
   C++ converts a pointer to a class into a pointer to one of its bases
   wherever one is wanted, and C makes you write the cast. The pass that writes
   it finds a call by the name of what it calls; a call through a table has no
   name, only the cast in front of it that says what each parameter is - so
   the derived pointer reached the C stage as it was, and was refused on a line
   that is correct C++.

   Three levels up, as COM's streams are, so that "a base" has to mean any
   base and not only the one written after the colon. And the initialiser on
   the next line, which is the same conversion in a place that already had
   it. */
#include <cstdio>

struct IUnknownish { virtual unsigned long AddRef() = 0; };
struct ISequential : IUnknownish { virtual int Read(int n) = 0; };
struct IStreamish : ISequential { virtual int Seek(int to) = 0; };
struct IWicStreamish : IStreamish { virtual int InitializeFrom(IStreamish *from) = 0; };
struct IEncoderish : IUnknownish { virtual int Initialize(IStreamish *into, int option) = 0; };

struct Stream : IWicStreamish {
    int at;
    Stream() : at(3) {}
    unsigned long AddRef() { return 1; }
    int Read(int n) { return at + n; }
    int Seek(int to) { at = to; return at; }
    int InitializeFrom(IStreamish *from) { return from->Seek(5); }
};

struct Encoder : IEncoderish {
    unsigned long AddRef() { return 1; }
    int Initialize(IStreamish *into, int option) { return into->Seek(40) + option; }
};

int main() {
    Stream s;
    Encoder e;
    IWicStreamish *stream = &s;
    IEncoderish *encoder = &e;
    int a = encoder->Initialize(stream, 2);
    ISequential *seq = stream;
    int b = seq->Read(1);
    std::printf("%d %d\n", a, b);
    return 0;
}
