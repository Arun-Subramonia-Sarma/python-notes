# One-liner UV Requirements

Quick one-liner commands to convert `requirements.txt` to UV dependencies.

## The One-liner

```bash
grep -v '^#' requirements.txt | grep -v '^$' | while read package; do uv add "$package"; done && uv sync
```

## What It Does

This single command:
1. `grep -v '^#'` - Removes lines starting with `#` (comments)
2. `grep -v '^$'` - Removes empty lines  
3. `while read package; do uv add "$package"; done` - Adds each package with `uv add`
4. `&& uv sync` - Syncs dependencies after all packages are added

## Prerequisites

- [UV](https://github.com/astral-sh/uv) installed
- `requirements.txt` file exists
- Run from your project directory

## Usage Examples

### Basic Usage
```bash
# Default requirements.txt
grep -v '^#' requirements.txt | grep -v '^$' | while read package; do uv add "$package"; done && uv sync
```

### Custom Requirements File
```bash
# Custom file
grep -v '^#' dev-requirements.txt | grep -v '^$' | while read package; do uv add "$package"; done && uv sync
```

### Add as Dev Dependencies
```bash
# Development dependencies
grep -v '^#' dev-requirements.txt | grep -v '^$' | while read package; do uv add --dev "$package"; done && uv sync
```

### With Error Handling
```bash
# Stop on first error
set -e && grep -v '^#' requirements.txt | grep -v '^$' | while read package; do uv add "$package"; done && uv sync
```

## Create a Reusable Alias

Add to your `~/.bashrc` or `~/.zshrc`:

```bash
# Simple alias
alias uv-requirements='grep -v "^#" requirements.txt | grep -v "^$" | while read package; do uv add "$package"; done && uv sync'

# Alias with parameter support
uv-add-from() {
    local file=${1:-requirements.txt}
    grep -v '^#' "$file" | grep -v '^$' | while read package; do uv add "$package"; done && uv sync
}

# Development dependencies alias
uv-add-dev() {
    local file=${1:-dev-requirements.txt}
    grep -v '^#' "$file" | grep -v '^$' | while read package; do uv add --dev "$package"; done && uv sync
}
```

Then use like:
```bash
# Use aliases
uv-requirements
uv-add-from custom-requirements.txt
uv-add-dev
```

## Create a Simple Script File

Save as `uv-req.sh`:
```bash
#!/bin/bash
file=${1:-requirements.txt}
grep -v '^#' "$file" | grep -v '^$' | while read package; do uv add "$package"; done && uv sync
```

Make executable and use:
```bash
chmod +x uv-req.sh
./uv-req.sh
./uv-req.sh dev-requirements.txt
```

## Variations

### With Progress Display
```bash
grep -v '^#' requirements.txt | grep -v '^$' | while read package; do echo "Adding: $package" && uv add "$package"; done && uv sync
```

### Count Packages First
```bash
total=$(grep -v '^#' requirements.txt | grep -v '^$' | wc -l) && echo "Adding $total packages..." && grep -v '^#' requirements.txt | grep -v '^$' | while read package; do uv add "$package"; done && uv sync
```

### With Error Recovery
```bash
grep -v '^#' requirements.txt | grep -v '^$' | while read package; do uv add "$package" || echo "Failed: $package"; done && uv sync
```

## Supported Requirements.txt Formats

Works with standard pip requirements:
```txt
# Comments are ignored
requests>=2.28.0
pandas==1.5.0
numpy  # Inline comments removed automatically

# Empty lines ignored

django>=4.0,<5.0
pytest
```

## When to Use

✅ **Perfect for:**
- Quick one-off conversions
- Terminal power users
- Automated scripts
- CI/CD pipelines

❌ **Consider full scripts for:**
- Error handling needs
- Progress reporting
- Development vs production separation
- Reusable workflows

## Integration Examples

### Makefile
```makefile
install:
	grep -v '^#' requirements.txt | grep -v '^$' | while read package; do uv add "$$package"; done && uv sync

install-dev:
	grep -v '^#' dev-requirements.txt | grep -v '^$' | while read package; do uv add --dev "$$package"; done && uv sync
```

### Package.json Scripts (if using Node.js too)
```json
{
  "scripts": {
    "install-py": "grep -v '^#' requirements.txt | grep -v '^$' | while read package; do uv add \"$package\"; done && uv sync"
  }
}
```

### Docker
```dockerfile
RUN grep -v '^#' requirements.txt | grep -v '^$' | while read package; do uv add "$package"; done && uv sync
```

## Common Issues & Solutions

| Issue | Solution |
|-------|----------|
| "No such file" | Check `requirements.txt` exists |
| "uv command not found" | Install UV first |
| Package fails to install | Remove problematic line, run again |
| Inline comments included | They're automatically stripped by the command |

## Why One-liners?

- ⚡ **Fast**: No script files to manage
- 🔧 **Flexible**: Easy to modify on the fly
- 📦 **Portable**: Works on any Unix system
- 🎯 **Focused**: Does one thing well

Perfect for developers who prefer command-line efficiency!