#!/bin/sh
# Differential test: the same C++ built two ways, and the outputs compared.
#
#   reference : clang++ - ground truth for what the C++ means
#   py2bin    : translated to C, then compiled by py2bin's own C compiler
#
# clang++ is the yardstick, not a dependency: py2bin builds with no toolchain.
# This asks whether the translation preserves meaning, which is the only thing
# reading the generated C cannot tell you.

# The clone this script is in, so it works wherever the clone is.
ROOT=$(cd "$(dirname "$0")/.." && pwd)

PASS=0; FAIL=0; REFUSED=0
for source in "$(dirname "$0")"/cpp_corpus/*.cpp; do
    name=$(basename "$source" .cpp)
    # c++03 first, because that is the shape of the subset; c++11 for the
    # programs whose feature only exists there - `char16_t` and `u"..."` are
    # not C++03 at all, so a reference for them has to be asked for in a
    # standard that has them.
    if ! clang++ -std=c++03 -w -o "/tmp/ref_$name" "$source" 2>/dev/null; then
        if ! clang++ -std=c++11 -w -o "/tmp/ref_$name" "$source" 2>/dev/null; then
            # <filesystem> is C++17, and its reference has to be asked for in
            # a standard that has it.
            if ! clang++ -std=c++17 -w -o "/tmp/ref_$name" "$source" 2>/dev/null; then
                printf "  %-28s reference did not build - skipped\n" "$name"
                continue
            fi
        fi
    fi
    want=$("/tmp/ref_$name" 2>&1); wantcode=$?
    got=$(PYTHONPATH="$ROOT/src" python3 -m py2bin cc \
          "$source" -o "/tmp/p2b_$name" 2>&1)
    if [ $? -ne 0 ]; then
        case "$got" in
            *"does not do"*|*"standard library"*|*"initialiser list"*|*"destructor"*)
                printf "  %-28s refused (says why)\n" "$name"
                REFUSED=$((REFUSED + 1)) ;;
            *)
                printf "  %-28s BUILD FAILED\n" "$name"
                printf "        %s\n" "$(printf '%s' "$got" | head -1 | cut -c1-96)"
                FAIL=$((FAIL + 1)) ;;
        esac
        continue
    fi
    mine=$("/tmp/p2b_$name" 2>&1); minecode=$?
    if [ "$mine" = "$want" ] && [ "$minecode" = "$wantcode" ]; then
        printf "  %-28s ok\n" "$name"
        PASS=$((PASS + 1))
    else
        printf "  %-28s DIFFERS\n" "$name"
        printf "        clang++: %s (exit %s)\n" "$(printf '%s' "$want" | tr '\n' '|' | cut -c1-70)" "$wantcode"
        printf "        py2bin : %s (exit %s)\n" "$(printf '%s' "$mine" | tr '\n' '|' | cut -c1-70)" "$minecode"
        FAIL=$((FAIL + 1))
    fi
done
echo
echo "  agreed $PASS   differed $FAIL   refused $REFUSED"
[ $FAIL -eq 0 ]
