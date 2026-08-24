#!/bin/sh
# Every C++ program in the corpus, put through everything py2bin can do to it.
#
#   tools/cpp_sweep.sh                    the whole sweep, over the corpus
#   tools/cpp_sweep.sh meaning            only the comparison against clang++
#   tools/cpp_sweep.sh targets            only the cross-build over all six
#   tools/cpp_sweep.sh check FILE [DIR]   the same two questions, asked of
#                                         *your* program rather than the
#                                         corpus. DIR is an include path.
#
# Two different questions, and both have to be asked:
#
#   meaning  - build it twice, once with clang++ and once with py2bin, run
#              both on this machine and compare. This is the only thing that
#              says the translation *means* the same; reading the generated C
#              tells you it is well formed and nothing else.
#   targets  - build it for every target py2bin supports. A construct can
#              translate perfectly and still fail to encode for one machine,
#              and this is the only thing that asks. It does not run them:
#              five of the six are not this computer.
#
# clang++ is the yardstick here and never a dependency - py2bin builds with
# no toolchain at all. Where clang++ is missing, `meaning` is skipped and
# `targets` still runs.
set -u
ROOT=$(cd "$(dirname "$0")/.." && pwd)
CORPUS="$ROOT/tools/cpp_corpus"
WHICH=${1:-all}
WORK=${TMPDIR:-/tmp}/py2bin-sweep
mkdir -p "$WORK"

TARGETS="darwin-arm64 darwin-x86_64 linux-x86_64 linux-arm64 windows-x86_64 windows-arm64"

agree=0; differ=0; refused=0; built=0; unbuilt=0

run_meaning() {
  command -v clang++ > /dev/null 2>&1 || {
    echo "  clang++ is not here, so there is nothing to compare against."
    return
  }
  echo "== meaning: py2bin against clang++, run on this machine =="
  for source in "$CORPUS"/*.cpp; do
    name=$(basename "$source" .cpp)
    std=""
    for s in c++03 c++11 c++17; do
      clang++ -std=$s -w -o "$WORK/ref_$name" "$source" 2>/dev/null && { std=$s; break; }
    done
    [ -z "$std" ] && { printf "  %-28s no reference - skipped\n" "$name"; continue; }
    want=$("$WORK/ref_$name" 2>&1); wantcode=$?
    got=$(PYTHONPATH="$ROOT/src" python3 -m py2bin cc "$source" -o "$WORK/p_$name" 2>&1)
    if [ $? -ne 0 ]; then
      printf "  %-28s REFUSED\n" "$name"
      printf '%s\n' "$got" | tail -1 | sed 's/^/        /'
      refused=$((refused + 1)); continue
    fi
    mine=$("$WORK/p_$name" 2>&1); minecode=$?
    if [ "$mine" = "$want" ] && [ "$minecode" = "$wantcode" ]; then
      agree=$((agree + 1))
    else
      printf "  %-28s DIFFERS\n" "$name"
      printf "        clang++: %s (exit %s)\n" "$(printf '%s' "$want" | tr '\n' '|')" "$wantcode"
      printf "        py2bin : %s (exit %s)\n" "$(printf '%s' "$mine" | tr '\n' '|')" "$minecode"
      differ=$((differ + 1))
    fi
  done
  printf "\n  agreed %d   differed %d   refused %d\n\n" "$agree" "$differ" "$refused"
}

run_targets() {
  echo "== targets: does each one encode for every machine =="
  for target in $TARGETS; do
    bad=""
    for source in "$CORPUS"/*.cpp; do
      name=$(basename "$source" .cpp)
      if PYTHONPATH="$ROOT/src" python3 -m py2bin cc "$source" \
           -o "$WORK/${target}_$name" --target "$target" > "$WORK/log" 2>&1; then
        built=$((built + 1))
      else
        unbuilt=$((unbuilt + 1))
        bad="$bad $name"
      fi
    done
    if [ -z "$bad" ]; then
      printf "  %-18s all built\n" "$target"
    else
      printf "  %-18s FAILED:%s\n" "$target" "$bad"
    fi
  done
  printf "\n  built %d   failed %d\n" "$built" "$unbuilt"
}

run_check() {
  entry=$1
  includes=${2:-}
  [ -f "$entry" ] || { echo "no such file: $entry"; exit 2; }
  name=$(basename "$entry" | sed 's/\.[^.]*$//')
  inc=""
  [ -n "$includes" ] && inc="--include-dir $includes"

  echo "== $entry: does it build for every machine =="
  for target in $TARGETS; do
    suffix=""
    case "$target" in windows-*) suffix=".exe" ;; esac
    if PYTHONPATH="$ROOT/src" python3 -m py2bin cc "$entry" $inc \
         -o "$WORK/${target}_$name$suffix" --target "$target" > "$WORK/log" 2>&1; then
      printf "  %-18s built\n" "$target"
      built=$((built + 1))
    else
      printf "  %-18s FAILED\n" "$target"
      sed 's/^/        /' "$WORK/log" | tail -3
      unbuilt=$((unbuilt + 1))
    fi
  done

  if command -v clang++ > /dev/null 2>&1; then
    echo ""
    echo "== $entry: does it mean what clang++ says it means =="
    std=""
    for s in c++03 c++11 c++17; do
      clang++ -std=$s -w -o "$WORK/ref_$name" "$entry" 2>/dev/null && { std=$s; break; }
    done
    if [ -z "$std" ]; then
      echo "  clang++ could not build it either, so there is nothing to compare."
    else
      want=$("$WORK/ref_$name" 2>&1); wantcode=$?
      host=$(PYTHONPATH="$ROOT/src" python3 -m py2bin targets 2>/dev/null | head -1)
      mine=$("$WORK/darwin-arm64_$name" 2>&1 || "$WORK/linux-x86_64_$name" 2>&1)
      minecode=$?
      if [ "$mine" = "$want" ]; then
        echo "  output matches clang++"
      else
        printf "  clang++: %s (exit %s)\n" "$(printf '%s' "$want" | tr '\n' '|')" "$wantcode"
        printf "  py2bin : %s (exit %s)\n" "$(printf '%s' "$mine" | tr '\n' '|')" "$minecode"
        differ=$((differ + 1))
      fi
    fi
  fi
  printf "\n  built %d   failed %d\n" "$built" "$unbuilt"
}

case "$WHICH" in
  meaning) run_meaning ;;
  targets) run_targets ;;
  check) run_check "${2:-}" "${3:-}" ;;
  *) run_meaning; run_targets ;;
esac

[ "$differ" -eq 0 ] && [ "$refused" -eq 0 ] && [ "$unbuilt" -eq 0 ]
