# UV Add Requirements Script

A shell script to read `requirements.txt` and add packages to your UV project using `uv add` commands.

## Features

- ✅ Reads packages from `requirements.txt` (or custom file)
- ✅ Uses `uv add` to properly manage dependencies in `pyproject.toml`
- ✅ Supports development dependencies with `--dev` flag
- ✅ Handles comments and empty lines in requirements files
- ✅ Dry run mode to preview changes
- ✅ Verbose output for debugging
- ✅ Automatic project initialization if needed
- ✅ Progress tracking and error reporting
- ✅ Automatic dependency syncing

## Prerequisites

- [UV package manager](https://github.com/astral-sh/uv) installed
- `requirements.txt` file in your project directory

### Install UV
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Installation

1. Download the script:
```bash
curl -O https://your-repo/uv_add_requirements.sh
chmod +x uv_add_requirements.sh
```

2. Or copy the script content to a file and make it executable:
```bash
chmod +x uv_add_requirements.sh
```

## Usage

### Basic Usage
```bash
# Add packages from requirements.txt
./uv_add_requirements.sh

# Add packages from custom requirements file
./uv_add_requirements.sh dev-requirements.txt
```

### Advanced Usage
```bash
# Add as development dependencies
./uv_add_requirements.sh --dev dev-requirements.txt

# Dry run - see what would be added without making changes
./uv_add_requirements.sh -d requirements.txt

# Verbose output
./uv_add_requirements.sh -v requirements.txt

# Show help
./uv_add_requirements.sh --help
```

## Command Line Options

| Option | Description |
|--------|-------------|
| `-h, --help` | Show help message |
| `-v, --verbose` | Enable verbose output |
| `-d, --dry-run` | Show what would be added without making changes |
| `--dev` | Add packages as development dependencies |

## Requirements.txt Format

The script supports standard `requirements.txt` format:

```txt
# Core dependencies
requests>=2.28.0
pandas==1.5.0
numpy

# Optional dependencies
pytest>=7.0.0  # Testing framework
black          # Code formatter

# Version ranges
django>=4.0,<5.0
```

### Supported formats:
- Package names: `requests`
- Version pinning: `pandas==1.5.0`
- Version ranges: `requests>=2.28.0`
- Complex versions: `django>=4.0,<5.0`
- Comments: `# This is a comment`
- Inline comments: `requests>=2.28.0  # HTTP library`

## What the Script Does

1. **Validates Environment**
   - Checks if `uv` is installed
   - Verifies requirements file exists
   - Initializes UV project if `pyproject.toml` doesn't exist

2. **Processes Requirements**
   - Parses `requirements.txt`
   - Skips comments and empty lines
   - Extracts clean package specifications

3. **Adds Dependencies**
   - Uses `uv add` for each package
   - Tracks success/failure for each package
   - Updates `pyproject.toml` automatically

4. **Syncs Project**
   - Runs `uv sync` to install dependencies
   - Creates/updates `uv.lock` file

## Example Output

```
uv version: uv 0.1.0
Found 5 packages to add
Adding packages using uv add...
================================
Adding: requests>=2.28.0
  ✅ Successfully added: requests>=2.28.0

Adding: pandas==1.5.0
  ✅ Successfully added: pandas==1.5.0

================================
📊 Summary:
  Total packages: 5
  Successfully added: 5
  Failed to add: 0
Installation completed at: Sun Aug 17 10:30:45 2025
Syncing dependencies...
✅ Dependencies synced successfully!
```

## Error Handling

The script handles various error scenarios:
- Missing requirements file
- UV not installed
- Invalid package specifications
- Network issues during installation
- Permission errors

Failed packages are reported in the summary with details about what went wrong.

## Project Structure After Running

```
your-project/
├── pyproject.toml      # Updated with new dependencies
├── uv.lock            # Locked dependency versions
├── requirements.txt   # Original requirements file
└── .venv/            # Virtual environment (created by uv sync)
```

## Integration with CI/CD

You can use this script in your CI/CD pipeline:

```yaml
# GitHub Actions example
- name: Setup UV
  run: curl -LsSf https://astral.sh/uv/install.sh | sh

- name: Add requirements
  run: ./uv_add_requirements.sh requirements.txt

- name: Run tests
  run: uv run pytest
```

## Troubleshooting

### Common Issues

1. **"uv not found"**
   - Install UV: `curl -LsSf https://astral.sh/uv/install.sh | sh`
   - Restart terminal or run: `source ~/.bashrc`

2. **"No pyproject.toml found"**
   - The script automatically initializes a UV project
   - Or manually run: `uv init`

3. **Package installation fails**
   - Check package name spelling
   - Verify version constraints are valid
   - Check network connectivity

4. **Permission denied**
   - Make script executable: `chmod +x uv_add_requirements.sh`

### Debug Mode

Run with verbose flag to see detailed output:
```bash
./uv_add_requirements.sh -v requirements.txt
```

## Contributing

Feel free to submit issues and enhancement requests!

## License

This script is provided as-is under the MIT License.