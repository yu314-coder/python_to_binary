#!/bin/sh
# The meaning half of the sweep, kept under its old name.
#
# Everything it did lives in tools/cpp_sweep.sh now, which asks the same
# question and one more - whether each program encodes for every target, not
# only for this machine. Two scripts over one corpus would drift.
exec "$(dirname "$0")/cpp_sweep.sh" meaning "$@"
