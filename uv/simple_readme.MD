# Simple UV Add Requirements Script

A lightweight shell script that reads `requirements.txt` and adds packages to your UV project using `uv add`.

## Overview

This is a minimal, no-frills script that:
- Reads packages from `requirements.txt`
- Adds each package using `uv add`
- Syncs dependencies with `uv sync`
- Handles basic error checking

Perfect for quick setups and simple automation.

## Prerequisites

- [UV package manager](https://github.com/astral-sh/uv) installed
- `requirements.txt` file

### Quick UV Installation
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Installation & Usage

1. **Download and make executable:**
```bash
curl -O https://your-repo/simple_uv_add.sh
chmod +x simple_uv_add.sh
```

2. **Run the script:**
```bash
# Use default requirements.txt
./simple_uv_add.sh

# Use custom requirements file
./simple_uv_add.sh dev-requirements.txt
```

## Script Behavior

```bash
./simple_uv_add.sh
```

**What it does:**
1. Checks if requirements file exists (defaults to `requirements.txt`)
2. Verifies UV is installed
3. Initializes UV project if no `pyproject.toml` exists
4. Reads each line from requirements file
5. Skips comments (`#`) and empty lines
6. Runs `uv add` for each package
7. Runs `uv sync` to install everything

## Example Output

```
Initializing uv project...
Adding packages from requirements.txt...
Adding: requests>=2.28.0
Adding: pandas==1.5.0
Adding: numpy
Syncing dependencies...
✅ All packages added successfully!
```

## Requirements.txt Format

Standard format supported:
```txt
# Web requests
requests>=2.28.0
urllib3

# Data processing
pandas==1.5.0
numpy

# Development tools
pytest  # Testing
```

## Error Handling

The script will exit with error messages for:
- Missing requirements file
- UV not installed
- Package installation failures

## File Structure After Running

```
your-project/
├── pyproject.toml      # Created/updated with dependencies
├── uv.lock            # Dependency lock file
├── requirements.txt   # Your original file
└── .venv/            # Virtual environment
```

## When to Use This Script

✅ **Good for:**
- Quick prototyping
- Simple projects
- Automated builds
- When you want minimal overhead

❌ **Consider the advanced script for:**
- Development vs production dependencies
- Verbose output needs
- Error recovery requirements
- Dry run capabilities

## Integration Examples

### Docker
```dockerfile
COPY requirements.txt simple_uv_add.sh ./
RUN chmod +x simple_uv_add.sh && ./simple_uv_add.sh
```

### GitHub Actions
```yaml
- name: Install dependencies
  run: |
    chmod +x simple_uv_add.sh
    ./simple_uv_add.sh
```

### Local Development
```bash
# Clone repo and setup
git clone your-repo
cd your-repo
./simple_uv_add.sh
uv run python main.py
```

## Troubleshooting

| Problem | Solution |
|---------|----------|
| "Permission denied" | `chmod +x simple_uv_add.sh` |
| "uv: command not found" | Install UV or restart terminal |
| "requirements.txt not found" | Create file or specify path |

## Script Source

```bash
#!/bin/bash
set -e
REQUIREMENTS_FILE="${1:-requirements.txt}"

[ ! -f "$REQUIREMENTS_FILE" ] && { echo "Error: $REQUIREMENTS_FILE not found!"; exit 1; }
command -v uv &> /dev/null || { echo "Error: uv not installed!"; exit 1; }

[ ! -f "pyproject.toml" ] && { echo "Initializing uv project..."; uv init --no-readme; }

echo "Adding packages from $REQUIREMENTS_FILE..."
while IFS= read -r line; do
    if [[ ! "$line" =~ ^[[:space:]]*# ]] && [[ -n "$line" ]]; then
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
```

## License

MIT License - use freely in your projects!