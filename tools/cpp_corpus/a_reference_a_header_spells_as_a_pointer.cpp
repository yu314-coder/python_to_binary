/* `factory->CreateEncoder(GUID_Jpeg, ...)` and `CoCreateInstance(CLSID_Thing,
   ...)` - a GUID handed to a parameter C++ declares as a reference. The
   Windows headers are written for both languages and say it once for each:
   `REFGUID` is `const GUID &` to C++ and `const GUID *` to C. py2bin reads the
   C branch, so both calls reached the C stage as a GUID where a pointer was
   wanted, and were refused on lines that are correct Windows C++.

   Two calls because they are two paths: a method reached through an
   interface's table, whose signature is written into the call itself, and a
   function a header declares. The first builds everywhere, because py2bin
   writes COM's shapes out itself; the second is an import only Windows has,
   so it is guarded - as a statement, which the translator keeps in place.

   clang++ has no <objbase.h> here, so there is no reference to compare with.
   What this asks is that it builds, on all six, and without the binding it
   does not: the C stage refuses the GUID. That the callee reads the object's
   own address is run in tests/test_c_frontend.py. */
#include <objbase.h>
#include <cstdio>

static const GUID CLSID_Nothing = {0x1, 0x2, 0x3, {0, 1, 2, 3, 4, 5, 6, 7}};
static const GUID IID_IFactoryish = {0x4, 0x5, 0x6, {7, 6, 5, 4, 3, 2, 1, 0}};
static const GUID GUID_Format = {0x19e4a5aa, 0x5662, 0x4fc5,
                                 {0xa0, 0xc0, 0x17, 0x58, 0x02, 0x8e, 0x10, 0x57}};

struct IEncoderish;

struct IFactoryish {
    virtual HRESULT STDMETHODCALLTYPE CreateEncoder(REFGUID format, const GUID *vendor,
                                                    IEncoderish **out) = 0;
};

struct Factory : IFactoryish {
    HRESULT STDMETHODCALLTYPE CreateEncoder(REFGUID format, const GUID *vendor,
                                            IEncoderish **out) {
        *out = 0;
        return vendor ? 1 : 2;
    }
};

int main() {
    Factory local;
    IFactoryish *factory = &local;
    IEncoderish *encoder = 0;
    HRESULT asked = factory->CreateEncoder(GUID_Format, 0, &encoder);
    printf("%d %d\n", (int)asked, encoder == 0);
#ifdef _WIN32
    IFactoryish *created = 0;
    HRESULT made = CoCreateInstance(CLSID_Nothing, 0, CLSCTX_INPROC_SERVER,
                                    IID_PPV_ARGS(&created));
    printf("%d\n", made == made);
#endif
    return 0;
}
