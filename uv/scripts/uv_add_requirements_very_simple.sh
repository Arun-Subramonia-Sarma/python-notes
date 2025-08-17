#!/bin/bash
# Convert requirements.txt to uv add commands
grep -v '^#' requirements.txt | grep -v '^$' | while read package; do uv add "$package"; done && uv sync