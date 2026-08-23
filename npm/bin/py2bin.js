#!/usr/bin/env node
"use strict";

// A wrapper, not a port. py2bin is a Python program that writes machine code
// with nothing but the standard library; this hands your arguments to it and
// gets out of the way, so `npx py2bin cc main.c util.c` works without anyone
// having to know that.
//
// What it does add is the part that goes wrong first: finding a Python new
// enough, and saying exactly what to run when py2bin is not installed yet.

const { spawnSync } = require("node:child_process");

const MINIMUM = [3, 10];
const CANDIDATES =
  process.platform === "win32"
    ? [["py", ["-3"]], ["python", []], ["python3", []]]
    : [["python3", []], ["python", []]];

// Ask an interpreter what it is. Printed rather than inferred from the name:
// `python3` is whatever is first on PATH, which is not always what it looks.
const PROBE =
  "import sys;print(sys.version_info[0], sys.version_info[1]);" +
  "import importlib.util as u;print(1 if u.find_spec('py2bin') else 0)";

function interrogate(command, prefix) {
  const asked = spawnSync(command, [...prefix, "-c", PROBE], {
    encoding: "utf8",
  });
  if (asked.status !== 0 || !asked.stdout) return null;
  const [version, installed] = asked.stdout.trim().split("\n");
  const [major, minor] = version.split(" ").map(Number);
  return { command, prefix, major, minor, installed: installed === "1" };
}

function newEnough(found) {
  return (
    found.major > MINIMUM[0] ||
    (found.major === MINIMUM[0] && found.minor >= MINIMUM[1])
  );
}

function main() {
  const seen = [];
  let usable = null;
  for (const [command, prefix] of CANDIDATES) {
    const found = interrogate(command, prefix);
    if (!found) continue;
    seen.push(`${command} (${found.major}.${found.minor})`);
    if (!newEnough(found)) continue;
    // Prefer one that already has py2bin; otherwise remember the first that
    // could have it, so the message below names an interpreter that exists.
    if (found.installed) {
      usable = found;
      break;
    }
    if (!usable) usable = found;
  }

  if (!usable) {
    console.error("py2bin needs Python 3.10 or newer, and none was found.");
    console.error(
      seen.length
        ? `  Found: ${seen.join(", ")} - all older than 3.10.`
        : "  Nothing named python3, python or py is on PATH."
    );
    console.error("  Install one from https://www.python.org/downloads/");
    return 1;
  }

  if (!usable.installed) {
    console.error("py2bin itself is not installed for that Python yet. Run:");
    console.error(
      `  ${usable.command} ${[...usable.prefix, "-m", "pip", "install", "python-to-binary"].join(" ")}`
    );
    console.error(
      "\n  This package is a wrapper: the compiler is the Python one, so it"
    );
    console.error("  has to be there for anything to be built.");
    return 1;
  }

  const run = spawnSync(
    usable.command,
    [...usable.prefix, "-m", "py2bin", ...process.argv.slice(2)],
    { stdio: "inherit" }
  );
  if (run.error) {
    console.error(`could not run ${usable.command}: ${run.error.message}`);
    return 1;
  }
  // A process killed by a signal has no exit code; report it the way a shell
  // does rather than as a silent success.
  if (run.signal) {
    console.error(`py2bin was killed by ${run.signal}`);
    return 1;
  }
  return run.status === null ? 1 : run.status;
}

process.exit(main());
