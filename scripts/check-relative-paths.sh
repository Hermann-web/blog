#!/bin/bash

BASE_DIR="docs"
MISSING=0

# Find all .md files
find "$BASE_DIR" -type f -name "*.md" | while read -r mdfile; do
    # Read file line by line with line numbers
    lineno=0
    while IFS= read -r line; do
        ((lineno++))
        # Extract all markdown links on this line
        # \[[^\]]+\]\(([^)]+)\)
        # Use grep -oP to get all links
        matches=$(echo "$line" | grep -oP '\[[^\]]+\]\(\K[^)]+(?=\))')
        for link in $matches; do
            # Skip if absolute URL or root path
            if [[ "$link" =~ ^https?:// ]] || [[ "$link" =~ ^/ ]]; then
                continue
            fi

            # Resolve path relative to mdfile directory
            link_path=$(realpath --no-symlinks --canonicalize-missing "$(dirname "$mdfile")/$link")

            if [ ! -e "$link_path" ]; then
                echo "$mdfile:$lineno Missing linked file: $link"
                MISSING=1
            fi
        done
    done < "$mdfile"
done

if [ "$MISSING" -eq 0 ]; then
    echo "✅ All relative paths in Markdown files exist."
else
    echo "⚠️ Some links are broken."
    exit 1
fi
