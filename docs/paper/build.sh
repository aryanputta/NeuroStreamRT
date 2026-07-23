#!/bin/sh
# Build a paper-kit paper to PDF with tectonic.
# Usage: ./build.sh <slug>        (defaults to the only .tex in this dir)
set -eu

if ! command -v tectonic >/dev/null 2>&1; then
  echo "tectonic not found. Install with: brew install tectonic" >&2
  exit 1
fi

slug="${1:-}"
if [ -z "$slug" ]; then
  set -- *.tex
  slug=$(basename "$1" .tex)
fi

tectonic "${slug}.tex"
echo "Built ${slug}.pdf"
