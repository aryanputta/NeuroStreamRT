#!/bin/sh
# Live edit loop: rebuild the SAME pdf in place whenever the .tex or a figure
# changes. Edit <slug>.tex in any editor; on save the PDF refreshes (macOS
# Preview reloads it automatically, so the open window updates with no new file).
#
# Usage: ./watch.sh <slug>      (defaults to the only .tex in this dir)
set -eu

if ! command -v tectonic >/dev/null 2>&1; then
  echo "tectonic not found. Install: brew install tectonic" >&2; exit 1
fi
if ! command -v fswatch >/dev/null 2>&1; then
  echo "fswatch not found. Install: brew install fswatch" >&2; exit 1
fi

slug="${1:-}"
if [ -z "$slug" ]; then
  set -- *.tex; slug=$(basename "$1" .tex)
fi

build() {
  if tectonic "${slug}.tex" >/tmp/paperbuild.log 2>&1; then
    printf "%s  rebuilt %s.pdf\n" "$(date +%H:%M:%S)" "$slug"
  else
    printf "%s  BUILD FAILED (see /tmp/paperbuild.log):\n" "$(date +%H:%M:%S)"
    tail -5 /tmp/paperbuild.log
  fi
}

echo "Watching ${slug}.tex and figs/ . Edit and save; Ctrl-C to stop."
[ -f "${slug}.pdf" ] && open "${slug}.pdf" 2>/dev/null || true
build
# watch the source and the figures dir; rebuild on any change
fswatch -o "${slug}.tex" figs 2>/dev/null | while read -r _; do
  build
done
