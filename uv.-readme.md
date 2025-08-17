# UV Projects and Virtual Environments - Complete Guide

A comprehensive guide to creating projects and managing virtual environments using UV, the fast Python package manager and project manager.

## Table of Contents

- [Installation](#installation)
- [Project Management](#project-management)
- [Virtual Environments](#virtual-environments)
- [Dependency Management](#dependency-management)
- [Running Python Code](#running-python-code)
- [Advanced Usage](#advanced-usage)
- [Migration from Other Tools](#migration-from-other-tools)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)

## Installation

### Install UV

```bash
# Using the official installer (recommended)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Using pip
pip install uv

# Using pipx
pipx install uv

# Using Homebrew (macOS)
brew install uv

# Using Conda
conda install -c conda-forge uv
```

### Verify Installation

```bash
uv --version
# Output: uv 0.4.0 (or your installed version)
```

## Project Management

### 1. Creating New Projects

#### Basic Project Creation
```bash
# Create a new project
uv init my-project
cd my-project

# Project structure created:
# my-project/
# ├── pyproject.toml
# ├── README.md
# ├── src/
# │   └── my_project/
# │       └── __init__.py
# └── .python-version
```

#### Create Project in Current Directory
```bash
# Initialize project in current directory
mkdir my-existing-project
cd my-existing-project
uv init
```

#### Create Project with Specific Python Version
```bash
# Create project with Python 3.11
uv init my-project --python 3.11

# Create project with specific Python executable
uv init my-project --python /usr/bin/python3.12
```

#### Create Different Project Types
```bash
# Create minimal project (no README, no src layout)
uv init my-project --no-readme --no-workspace

# Create library project
uv init my-library --lib

# Create application project
uv init my-app --app
```

### 2. Project Structure Explained

After running `uv init my-project`, you get:

```
my-project/
├── pyproject.toml      # Project configuration and dependencies
├── README.md           # Project documentation
├── src/                # Source code directory
│   └── my_project/     # Your package
│       └── __init__.py # Package initialization
├── .python-version     # Python version specification
└── uv.lock            # Locked dependency versions (created after first install)
```

#### pyproject.toml Structure
```toml
[project]
name = "my-project"
version = "0.1.0"
description = "Add your description here"
readme = "README.md"
requires-python = ">=3.8"
dependencies = []

[project.scripts]
my-project = "my_project:main"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

## Virtual Environments

### 1. Creating Virtual Environments

#### Basic Virtual Environment
```bash
# Create virtual environment in current directory
uv venv

# Creates .venv/ directory with Python environment
```

#### Custom Virtual Environment Location
```bash
# Create virtual environment with custom name
uv venv my-custom-env

# Create virtual environment in specific location
uv venv /path/to/my-env

# Create with specific Python version
uv venv --python 3.11
uv venv --python python3.12
```

#### Virtual Environment Options
```bash
# Create with system site packages access
uv venv --system-site-packages

# Create with specific prompt name
uv venv --prompt my-project

# Seed with pip (if needed for compatibility)
uv venv --seed
```

### 2. Using Virtual Environments

#### Activation (Manual)
```bash
# On Linux/macOS
source .venv/bin/activate

# On Windows
.venv\Scripts\activate

# On Windows (PowerShell)
.venv\Scripts\Activate.ps1
```

#### UV Managed Execution (Recommended)
```bash
# Run commands in the virtual environment without activation
uv run python script.py
uv run python -c "import sys; print(sys.executable)"
uv run pip list

# Install and run in one command
uv run --with requests python -c "import requests; print(requests.get('https://httpbin.org/json').json())"
```

### 3. Virtual Environment Management

#### List Environments
```bash
# Show current environment info
uv venv --show-path

# List all Python installations UV can use
uv python list
```

#### Remove Environments
```bash
# Remove .venv directory
rm -rf .venv

# Or on Windows
rmdir /s .venv
```

## Dependency Management

### 1. Adding Dependencies

#### Add Runtime Dependencies
```bash
# Add single package
uv add requests

# Add multiple packages
uv add requests pandas numpy

# Add with version constraints
uv add "requests>=2.28.0"
uv add "django>=4.0,<5.0"

# Add from PyPI with extras
uv add "fastapi[all]"
```

#### Add Development Dependencies
```bash
# Add development dependencies
uv add --dev pytest black isort

# Add with group specification
uv add --group test pytest pytest-cov
uv add --group lint black isort mypy
```

#### Add from Different Sources
```bash
# Add from Git repository
uv add git+https://github.com/user/repo.git

# Add from Git with specific branch/tag
uv add git+https://github.com/user/repo.git@main
uv add git+https://github.com/user/repo.git@v1.0.0

# Add from local path
uv add ./local-package
uv add ../my-other-project

# Add from URL
uv add https://files.pythonhosted.org/packages/.../package.whl
```

### 2. Installing Dependencies

#### Install All Dependencies
```bash
# Install all dependencies (creates uv.lock if not exists)
uv sync

# Install only production dependencies
uv sync --no-dev

# Install specific group
uv sync --group test
```

#### Install from Lock File
```bash
# Install exact versions from uv.lock
uv sync --frozen

# Install without updating lock file
uv sync --locked
```

### 3. Managing Dependencies

#### Remove Dependencies
```bash
# Remove package
uv remove requests

# Remove development dependency
uv remove --dev pytest

# Remove from specific group
uv remove --group test pytest
```

#### Update Dependencies
```bash
# Update all dependencies
uv sync --upgrade

# Update specific package
uv add requests --upgrade

# Update to latest compatible versions
uv lock --upgrade
```

#### List Dependencies
```bash
# Show dependency tree
uv tree

# List installed packages
uv pip list

# Show outdated packages
uv pip list --outdated
```

## Running Python Code

### 1. Project Execution

#### Run Python Scripts
```bash
# Run script with project dependencies
uv run python my_script.py

# Run module
uv run python -m my_project

# Run with additional dependencies
uv run --with beautifulsoup4 python scraper.py
```

#### Run Project Scripts
```bash
# If you have scripts defined in pyproject.toml
uv run my-project

# Run entry points
uv run my-command
```

### 2. Interactive Python

```bash
# Start Python REPL with project dependencies
uv run python

# Start IPython if available
uv run ipython

# Start with additional packages
uv run --with ipython ipython
```

### 3. Running Tools

```bash
# Run formatters
uv run black .
uv run isort .

# Run linters
uv run flake8 src/
uv run mypy src/

# Run tests
uv run pytest
uv run pytest tests/ -v

# Run with coverage
uv run coverage run -m pytest
uv run coverage report
```

## Advanced Usage

### 1. Workspace Management

#### Multi-package Workspaces
```bash
# Create workspace
mkdir my-workspace
cd my-workspace
uv init --workspace

# Add packages to workspace
mkdir packages/web-app packages/shared-lib
uv init packages/web-app
uv init packages/shared-lib --lib
```

#### Workspace Configuration
```toml
# pyproject.toml (workspace root)
[tool.uv.workspace]
members = ["packages/*"]

[tool.uv]
dev-dependencies = [
    "pytest>=7.0.0",
    "black>=22.0.0",
]
```

### 2. Environment Variables and Configuration

#### UV Configuration
```bash
# Set UV cache directory
export UV_CACHE_DIR=/path/to/cache

# Set Python installation directory
export UV_PYTHON_INSTALL_DIR=/path/to/pythons

# Disable UV from managing Python installations
export UV_PYTHON_DOWNLOADS=never
```

#### Project Configuration
```toml
# pyproject.toml
[tool.uv]
dev-dependencies = [
    "pytest>=7.0.0",
    "black>=22.0.0",
]

# Environment variables for scripts
[tool.uv.scripts]
test = "pytest tests/"
lint = { cmd = "black --check .", env = { PYTHONPATH = "src" } }
```

### 3. Docker Integration

#### Dockerfile with UV
```dockerfile
FROM python:3.11-slim

# Install UV
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Copy project files
COPY pyproject.toml uv.lock ./

# Install dependencies
RUN uv sync --frozen --no-cache

# Copy source code
COPY . .

# Run application
CMD ["uv", "run", "python", "-m", "my_project"]
```

#### Multi-stage Build
```dockerfile
FROM python:3.11-slim as builder

COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-cache

FROM python:3.11-slim

COPY --from=builder /.venv /.venv
COPY . .

ENV PATH="/.venv/bin:$PATH"
CMD ["python", "-m", "my_project"]
```

## Migration from Other Tools

### 1. From requirements.txt

#### Using UV Scripts (from previous READMEs)
```bash
# Use the advanced script
./uv_add_requirements.sh requirements.txt

# Use simple script
./simple_uv_add.sh requirements.txt

# Use one-liner
grep -v '^#' requirements.txt | grep -v '^$' | while read package; do uv add "$package"; done && uv sync
```

#### Manual Conversion
```bash
# Initialize UV project
uv init

# Add each dependency
uv add $(cat requirements.txt | grep -v '^#' | grep -v '^$' | tr '\n' ' ')

# Or add development dependencies
uv add --dev $(cat dev-requirements.txt | grep -v '^#' | grep -v '^$' | tr '\n' ' ')
```

### 2. From Poetry

#### Convert pyproject.toml
```bash
# UV can often work with existing pyproject.toml from Poetry
cd existing-poetry-project

# Install with UV
uv sync

# Or reinitialize if needed
uv init --no-readme
# Then manually copy dependencies from [tool.poetry.dependencies]
```

### 3. From Pipenv

#### Convert Pipfile
```bash
# Install pipfile-requirements first
pip install pipfile-requirements

# Convert Pipfile to requirements.txt
pipfile2req > requirements.txt

# Then use UV scripts to add dependencies
uv init
./uv_add_requirements.sh requirements.txt
```

### 4. From Conda

#### Convert environment.yml
```bash
# Extract pip dependencies from environment.yml
grep -A 100 "pip:" environment.yml | grep -E "^\s+-" | sed 's/^\s*-\s*//' > requirements.txt

# Initialize UV project and add dependencies
uv init
./uv_add_requirements.sh requirements.txt
```

## Best Practices

### 1. Project Organization

```bash
# Recommended project structure
my-project/
├── src/
│   └── my_project/         # Main package
│       ├── __init__.py
│       ├── cli.py          # CLI interface
│       ├── core/           # Core business logic
│       └── utils/          # Utilities
├── tests/                  # Test files
│   ├── __init__.py
│   ├── conftest.py         # Pytest configuration
│   └── test_my_project.py
├── docs/                   # Documentation
├── scripts/                # Development scripts
├── pyproject.toml          # Project configuration
├── uv.lock                # Locked dependencies
├── README.md
└── .gitignore
```

### 2. Dependency Management Best Practices

#### Version Pinning Strategy
```bash
# Pin major versions for stability
uv add "requests>=2.28.0,<3.0.0"

# Pin exact versions for security-critical packages
uv add "cryptography==41.0.7"

# Use compatible release specifier
uv add "django~=4.2.0"  # Equivalent to >=4.2.0, ==4.2.*
```

#### Dependency Groups
```bash
# Organize dependencies by purpose
uv add --group test pytest pytest-cov pytest-mock
uv add --group lint black isort flake8 mypy
uv add --group docs sphinx sphinx-rtd-theme
uv add --group dev pre-commit tox
```

### 3. CI/CD Configuration

#### GitHub Actions
```yaml
name: Test

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.9", "3.10", "3.11", "3.12"]
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up uv
      run: curl -LsSf https://astral.sh/uv/install.sh | sh
    
    - name: Set up Python ${{ matrix.python-version }}
      run: uv python install ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: uv sync --all-extras --dev
    
    - name: Run tests
      run: uv run pytest
    
    - name: Run linting
      run: |
        uv run black --check .
        uv run isort --check-only .
        uv run flake8 .
```

#### Pre-commit Configuration
```yaml
# .pre-commit-config.yaml
repos:
  - repo: local
    hooks:
      - id: black
        name: black
        entry: uv run black
        language: system
        types: [python]
      
      - id: isort
        name: isort
        entry: uv run isort
        language: system
        types: [python]
      
      - id: pytest
        name: pytest
        entry: uv run pytest
        language: system
        pass_filenames: false
        always_run: true
```

### 4. Development Workflow

#### Daily Commands
```bash
# Start working on project
cd my-project
uv sync  # Ensure dependencies are up to date

# Add new dependency
uv add new-package

# Run tests
uv run pytest

# Format code
uv run black .
uv run isort .

# Type checking
uv run mypy src/

# Run application
uv run python -m my_project
```

#### Script Shortcuts
Add to pyproject.toml:
```toml
[tool.uv.scripts]
test = "pytest tests/"
lint = ["black --check .", "isort --check-only .", "flake8 ."]
format = ["black .", "isort ."]
typecheck = "mypy src/"
serve = { cmd = "python -m my_project.server", env = { DEBUG = "1" } }
```

Then use:
```bash
uv run test
uv run lint
uv run format
uv run typecheck
uv run serve
```

## Troubleshooting

### Common Issues and Solutions

#### 1. Python Version Issues
```bash
# Problem: Wrong Python version
# Solution: Specify Python version explicitly
uv venv --python 3.11

# List available Python versions
uv python list

# Install specific Python version
uv python install 3.11
```

#### 2. Package Installation Failures
```bash
# Problem: Package not found
# Solution: Check package name and availability
uv add --dry-run package-name

# Problem: Dependency conflicts
# Solution: Check dependency tree
uv tree
```

#### 3. Virtual Environment Issues
```bash
# Problem: Environment not activated
# Solution: Use uv run instead of manual activation
uv run python script.py

# Problem: Old environment causing issues
# Solution: Recreate environment
rm -rf .venv
uv sync
```

#### 4. Lock File Issues
```bash
# Problem: Lock file out of sync
# Solution: Regenerate lock file
rm uv.lock
uv sync

# Problem: Can't reproduce exact environment
# Solution: Use frozen install
uv sync --frozen
```

### Debug Mode

```bash
# Enable verbose output
uv -v add package-name
uv -v sync

# Show what UV would do without doing it
uv add --dry-run package-name
```

### Performance Tips

```bash
# Use UV cache effectively
export UV_CACHE_DIR=/path/to/fast/storage

# Parallel installation (default)
uv sync  # UV installs packages in parallel by default

# Skip expensive operations when possible
uv sync --no-build-isolation  # Skip build isolation if not needed
```

## Resources and Links

- **Official Documentation**: https://docs.astral.sh/uv/
- **GitHub Repository**: https://github.com/astral-sh/uv
- **Migration Guide**: https://docs.astral.sh/uv/guides/migration/
- **PyPI Package**: https://pypi.org/project/uv/

## Quick Reference Card

```bash
# Project Management
uv init my-project          # Create new project
uv init                     # Initialize in current directory

# Virtual Environments
uv venv                     # Create virtual environment
uv venv --python 3.11       # Create with specific Python

# Dependencies
uv add package              # Add dependency
uv add --dev package        # Add dev dependency
uv remove package           # Remove dependency
uv sync                     # Install all dependencies

# Running Code
uv run python script.py     # Run Python script
uv run pytest              # Run tests
uv run --with pkg python    # Run with temporary dependency

# Information
uv tree                     # Show dependency tree
uv pip list                 # List installed packages
uv --version               # Show UV version
```

This guide covers everything you need to know about using UV for Python project and virtual environment management!