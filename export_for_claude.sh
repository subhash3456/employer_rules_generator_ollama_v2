#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_FILE="${1:-"$ROOT_DIR/claude_context.txt"}"

case "$OUTPUT_FILE" in
  /*) ;;
  *) OUTPUT_FILE="$ROOT_DIR/$OUTPUT_FILE" ;;
esac

mkdir -p -- "$(dirname -- "$OUTPUT_FILE")"
: > "$OUTPUT_FILE"

{
  printf '===== PROJECT DIRECTORIES =====\n'
  find "$ROOT_DIR" \
    \( -type d \( \
      -name .git -o \
      -name .venv -o \
      -name venv -o \
      -name node_modules -o \
      -name __pycache__ -o \
      -name .pytest_cache \
    \) -prune \) -o \
    \( -mindepth 1 -type d -print \) |
    sort |
    while IFS= read -r directory; do
      printf '%s\n' "${directory#"$ROOT_DIR"/}"
    done
} >> "$OUTPUT_FILE"

find "$ROOT_DIR" \
  \( -type d \( \
    -name .git -o \
    -name .venv -o \
    -name venv -o \
    -name node_modules -o \
    -name __pycache__ -o \
    -name .pytest_cache -o \
    -name output \
  \) -prune \) -o \
  \( -type f \
    \( -name '*.py' -o -name '*.json' -o -name '*.md' -o \
       -name '*.txt' -o -name '*.toml' -o -name '*.yaml' -o \
       -name '*.yml' -o -name '*.sh' \) \
    ! -name '.env*' \
    ! -name 'claude_context.txt' \
    ! -name 'export_for_claude.sh' \
    -print \) |
  sort |
  while IFS= read -r file; do
    {
      printf '\n\n===== %s =====\n\n' "${file#"$ROOT_DIR"/}"
      cat -- "$file"
    } >> "$OUTPUT_FILE"
  done

printf 'Claude context bundle written to: %s\n' "$OUTPUT_FILE"
