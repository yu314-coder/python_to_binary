export const meta = {
  name: 'cpp-hunt',
  description: 'Hunt for C++-to-C translation bugs by probing every corner of the language, building through build.py',
  whenToUse: 'Before shipping a C++ project through py2bin, or after changing the C++ front end. Finds what the corpus does not cover yet.',
  phases: [
    { title: 'Probe', detail: 'one agent per area of C++, writing programs and building them through build.py' },
    { title: 'Verify', detail: 'each reported failure reduced to a minimal repro and confirmed' },
    { title: 'Report', detail: 'the confirmed failures, ordered by how likely a real program is to hit them' },
  ],
}

// Areas of the language, one agent each. Split so that two agents rarely
// write the same program: overlap wastes a slot and finds nothing new.
const AREAS = [
  { key: 'classes',    what: 'classes, constructors, destructors, member access, const methods, static members and member functions, nested classes, friend, this, copy construction, assignment operators, member initialiser lists, default arguments' },
  { key: 'inheritance', what: 'single inheritance, virtual functions, abstract classes, virtual destructors, calling a base implementation, upcasting through pointers and references, and what happens with several levels' },
  { key: 'templates',  what: 'class and function templates, several type parameters, templates that use other templates, overloaded templates, deduction from literals, variables, calls and expressions, and explicit instantiation' },
  { key: 'lambdas',    what: 'lambdas with and without captures, by value and by reference, [=] and [&], explicit and deduced return types, lambdas as comparators and as arguments, and function objects with operator()' },
  { key: 'memory',     what: 'new and delete, new[] and delete[], objects on the heap, pointers to objects, arrays of objects, arrays of pointers, pointer arithmetic, and destructors running when they should' },
  { key: 'exceptions', what: 'throw and catch of numbers and of objects, catch by reference, nested try, rethrowing, exceptions crossing several functions, destructors during unwinding, and what a program does when nothing catches' },
  { key: 'stdlib',     what: "py2bin's own <string>, <vector>, <iostream>, <algorithm>, <functional>, <stdexcept>, <utility>, <numeric>, <filesystem>, and the C headers: <string.h>, <ctype.h>, <math.h>, <stdlib.h>, <assert.h>" },
  { key: 'syntax',     what: 'enums plain and scoped, unions, range-based for, auto, bool/true/false/nullptr, named casts, switch, do-while, goto, bit operations, the ternary operator, function pointers, forward declarations, and multi-declarator statements like `int a = 1, b = 2;`' },
  { key: 'text',       what: 'string literals with non-ASCII text, wide and Unicode literals (L, u, U, u8), escapes, character constants, printf conversions, and text handling generally' },
  { key: 'mixed',      what: 'a C++ file compiled together with C files and its own headers, headers included more than once, classes declared in a header and used from a source file, and namespaces including aliases' },
]

const FINDING = {
  type: 'object',
  properties: {
    failures: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          title: { type: 'string', description: 'what does not work, in a few words' },
          source: { type: 'string', description: 'the smallest complete .cpp that shows it' },
          expected: { type: 'string', description: "what clang++ printed, or 'builds' if clang++ built it and py2bin did not" },
          got: { type: 'string', description: 'what py2bin printed, or the error it gave' },
          kind: { type: 'string', enum: ['refused', 'differs', 'crashes', 'target-only'] },
          target: { type: 'string', description: 'which target, where it is one target only' },
        },
        required: ['title', 'source', 'expected', 'got', 'kind'],
      },
    },
  },
  required: ['failures'],
}

const VERDICT = {
  type: 'object',
  properties: {
    real: { type: 'boolean', description: 'whether it reproduces from a clean state' },
    why: { type: 'string', description: 'one sentence: what actually happens' },
    likelihood: { type: 'string', enum: ['common', 'occasional', 'rare'], description: 'how likely a real C++ program is to write this' },
  },
  required: ['real', 'why', 'likelihood'],
}

const ROOT = '/Volumes/D/github/python_to_binary'

const HOW = `
Build **only** through the entry point users use:

    python3 ${ROOT}/build.py PROGRAM.cpp --target TARGET

It writes to \`dist/\` beside the program. Do not use \`python -m py2bin\`:
the point is to check the path people actually take. Targets are
darwin-arm64, darwin-x86_64, linux-x86_64, linux-arm64, windows-x86_64
and windows-arm64; only darwin-arm64 binaries can be *run* on this
machine, so for the others "does it build" is the whole question.

Compare against clang++ for meaning:

    clang++ -std=c++17 -w -o /tmp/ref PROGRAM.cpp && /tmp/ref

clang++ is the yardstick and never a dependency. Where clang++ cannot
build your program either, the program is wrong and not py2bin.

Work in a scratch directory of your own under /tmp. Do not edit anything
under ${ROOT}.
`

phase('Probe')
const found = await parallel(AREAS.map((area) => () =>
  agent(
    `You are hunting for bugs in py2bin's C++-to-C translator, in this area:\n\n` +
    `  ${area.what}\n\n` +
    `Write **at least 12** small, complete C++ programs that exercise it, each ` +
    `printing something to stdout so its behaviour can be compared. Prefer what ` +
    `a real program would write over what a language lawyer would.\n\n${HOW}\n\n` +
    `First check ${ROOT}/tools/cpp_corpus/ - a program equivalent to one already ` +
    `there is a wasted slot. Report only what actually failed, with the smallest ` +
    `source that still shows it. If everything you tried worked, report no ` +
    `failures; that is a useful answer and you should not invent one.`,
    { label: `probe:${area.key}`, phase: 'Probe', schema: FINDING },
  ),
))

const raw = found.filter(Boolean).flatMap((r) => r.failures || [])
log(`${raw.length} candidate failures from ${AREAS.length} areas`)

// Deduped against each other before anything expensive: several areas
// touching one bug is the common case, and verifying it five times is five
// times the cost for one answer.
const seen = new Set()
const unique = raw.filter((f) => {
  const key = `${f.kind}:${f.title.toLowerCase().replace(/[^a-z ]/g, '').slice(0, 40)}`
  if (seen.has(key)) return false
  seen.add(key)
  return true
})
log(`${unique.length} after removing duplicates`)

phase('Verify')
const judged = await parallel(unique.map((f) => () =>
  agent(
    `Check whether this is a real py2bin bug. Try to show it is *not* one - ` +
    `a mistake in the test program, a construct clang++ also rejects, or ` +
    `something already fixed. Set real=false unless it reproduces.\n\n` +
    `Reported: ${f.title}\nExpected: ${f.expected}\nGot: ${f.got}\n\n` +
    `Source:\n\`\`\`cpp\n${f.source}\n\`\`\`\n\n${HOW}`,
    { label: `verify:${f.title.slice(0, 28)}`, phase: 'Verify', schema: VERDICT },
  ).then((v) => ({ ...f, verdict: v })),
))

const confirmed = judged.filter(Boolean).filter((f) => f.verdict?.real)
const order = { common: 0, occasional: 1, rare: 2 }
confirmed.sort((a, b) => order[a.verdict.likelihood] - order[b.verdict.likelihood])

phase('Report')
log(`${confirmed.length} confirmed of ${unique.length} checked`)
return {
  confirmed: confirmed.map((f) => ({
    title: f.title,
    kind: f.kind,
    likelihood: f.verdict.likelihood,
    why: f.verdict.why,
    target: f.target || 'all',
    source: f.source,
    expected: f.expected,
    got: f.got,
  })),
  checked: unique.length,
  areas: AREAS.length,
}
