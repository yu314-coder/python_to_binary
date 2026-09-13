/* `IID_PPV_ARGS(&factory)` - one name that stands for two arguments, and
   which two depends on the type of what it is handed. The Windows SDK writes
   it `__uuidof(**(pp)), IID_PPV_ARGS_Helper(pp)`, which asks the compiler
   what `pp` points at; a macro cannot answer that, and py2bin's `__uuidof`
   takes the name of an interface rather than an expression. So the call
   reached the C stage with one argument where the API wants two, reported as
   a count that did not match on a line that is correct Windows C++.

   Read here instead, from the declaration of what is handed to it. The
   reference build spells the same thing with an overload, which is how C++
   answers the question - so the two agree only if the deduction does. Two
   interfaces, because a single answer written into the macro would get one
   of them right by luck. */
#include <cstdio>

struct Identifier { unsigned long value; };
typedef Identifier IID;

struct IThing { int held; };
struct IOther { int held; };

static IID IID_IThing = {11};
static IID IID_IOther = {22};

static const IID *chosen_for(IThing **) { return &IID_IThing; }
static const IID *chosen_for(IOther **) { return &IID_IOther; }

/* Only the reference build reads this: py2bin rewrites the call before any
   preprocessor sees it. */
#ifndef IID_PPV_ARGS
#define IID_PPV_ARGS(pp) chosen_for(pp), (void **)(pp)
#endif

static IThing one_thing;
static IOther one_other;

static int Create(unsigned long which, const IID *id, void **out) {
    if (id->value == 11) { one_thing.held = (int)(which + id->value); *out = &one_thing; }
    else { one_other.held = (int)(which + id->value); *out = &one_other; }
    return (int)id->value;
}

int main() {
    IThing *thing = 0;
    IOther *other = 0;
    int first = Create(1, IID_PPV_ARGS(&thing));
    int second = Create(2, IID_PPV_ARGS(&other));
    std::printf("%d %d %d %d\n", first, second,
                thing ? thing->held : -1, other ? other->held : -1);
    return 0;
}
