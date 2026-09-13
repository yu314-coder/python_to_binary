/* A struct handed over by value through a pointer declared __stdcall, to a
   function declared __stdcall as well - the shape of a method on a table a
   Windows header declares. Windows passes a struct this size as the address
   of a copy the callee owns, and py2bin's own calls pass an address the
   callee copies from; either way the callee has an object of its own, and
   writing to it changes nothing the caller holds. clang ignores the word off
   Windows, and so does py2bin. */
#include <stdio.h>

typedef struct Span {
    long long first;
    long long second;
    long long third;
} Span;

typedef struct Table Table;
struct Table {
    long long (__stdcall *total)(Table *self, Span span);
    int (__stdcall *count)(Table *self, int more);
    int seen;
};

static long long __stdcall add_up(Table *self, Span span) {
    long long sum = span.first + span.second + span.third;
    span.first = -1;
    span.third = -3;
    self->seen += 1;
    return sum + span.first + span.third + 4;
}

static int __stdcall count_up(Table *self, int more) {
    self->seen += more;
    return self->seen;
}

int main(void) {
    Table table;
    Span held;
    long long answer;
    table.total = add_up;
    table.count = count_up;
    table.seen = 0;
    held.first = 11;
    held.second = 22;
    held.third = 33;
    answer = table.total(&table, held);
    printf("%lld %lld %lld %lld\n", answer, held.first, held.second, held.third);
    printf("%d\n", table.count(&table, 5));
    return 0;
}
