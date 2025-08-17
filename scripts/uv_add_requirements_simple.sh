#!/bin/bash

# simple_uv_add.sh
# Simple script to add packages from requirements.txt using uv add

set -e

REQUIREMENTS_FILE="${1:-requirements.txt}"

# Check requirements file exists
if [ ! -f "$REQUIREMENTS_FILE" ]; then
    echo "Error: $REQUIREMENTS_FILE not found!"
    exit 1
fi

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "Error: uv not installed!"
    exit 1
fi

# Initialize uv project if needed
if [ ! -f "pyproject.toml" ]; then
    echo "Initializing uv project..."
    uv init --no-readme
fi

echo "Adding packages from $REQUIREMENTS_FILE..."

# Read each line and add with uv add
while IFS= read -r line; do
    # Skip comments and empty lines
    if [[ ! "$line" =~ ^[[:space:]]*# ]] && [[ -n "$line" ]]; then
        # Clean the line (remove inline comments and whitespace)
        clean_line=$(echo "$line" | sed 's/#.*//' | xargs)
        if [[ -n "$clean_line" ]]; then
            echo "Adding: $clean_line"
            uv add "$clean_line"
        fi
    fi
done < "$REQUIREMENTS_FILE"

echo "Syncing dependencies..."
uv sync

echo "✅ All packages added successfully!"