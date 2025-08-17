#!/bin/bash

# uv_add_requirements.sh
# Script to add packages from requirements.txt using uv add

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
REQUIREMENTS_FILE="requirements.txt"
VERBOSE=false
DRY_RUN=false
DEV_DEPS=false

# Function to print colored output
print_color() {
    printf "${1}${2}${NC}\n"
}

# Function to show usage
show_usage() {
    echo "Usage: $0 [OPTIONS] [requirements_file]"
    echo ""
    echo "OPTIONS:"
    echo "  -h, --help          Show this help message"
    echo "  -v, --verbose       Enable verbose output"
    echo "  -d, --dry-run       Show what would be added without adding"
    echo "  --dev               Add packages as development dependencies"
    echo ""
    echo "EXAMPLES:"
    echo "  $0                          # Add from requirements.txt"
    echo "  $0 dev-requirements.txt     # Add from custom file"
    echo "  $0 --dev dev-requirements.txt  # Add as dev dependencies"
    echo "  $0 -v -d requirements.txt   # Verbose dry run"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_usage
            exit 0
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -d|--dry-run)
            DRY_RUN=true
            shift
            ;;
        --dev)
            DEV_DEPS=true
            shift
            ;;
        -*)
            print_color $RED "Error: Unknown option $1"
            show_usage
            exit 1
            ;;
        *)
            REQUIREMENTS_FILE="$1"
            shift
            ;;
    esac
done

# Check if requirements file exists
if [ ! -f "$REQUIREMENTS_FILE" ]; then
    print_color $RED "Error: Requirements file '$REQUIREMENTS_FILE' not found!"
    show_usage
    exit 1
fi

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    print_color $RED "Error: uv is not installed or not in PATH"
    print_color $YELLOW "Install uv with: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

# Check if we're in a uv project (has pyproject.toml)
if [ ! -f "pyproject.toml" ]; then
    print_color $YELLOW "Warning: No pyproject.toml found. Initializing uv project..."
    if [ "$DRY_RUN" = false ]; then
        uv init --no-readme
        print_color $GREEN "✅ Initialized uv project"
    else
        print_color $YELLOW "[DRY RUN] Would initialize uv project"
    fi
fi

# Show uv version
print_color $BLUE "uv version: $(uv --version)"

# Read and process requirements.txt
print_color $BLUE "Reading packages from $REQUIREMENTS_FILE..."

# Array to store packages
declare -a packages=()
declare -a failed_packages=()

# Read requirements file and extract package names
while IFS= read -r line; do
    # Skip comments and empty lines
    if [[ "$line" =~ ^[[:space:]]*# ]] || [[ -z "$line" ]]; then
        continue
    fi
    
    # Remove inline comments and whitespace
    clean_line=$(echo "$line" | sed 's/#.*//' | xargs)
    
    if [[ -n "$clean_line" ]]; then
        packages+=("$clean_line")
    fi
done < "$REQUIREMENTS_FILE"

# Count total packages
TOTAL_PACKAGES=${#packages[@]}
print_color $BLUE "Found $TOTAL_PACKAGES packages to add"

# Show what would be added if dry run
if [ "$DRY_RUN" = true ]; then
    print_color $YELLOW "DRY RUN - Packages that would be added:"
    print_color $YELLOW "========================================"
    for package in "${packages[@]}"; do
        if [ "$DEV_DEPS" = true ]; then
            print_color $YELLOW "  uv add --dev '$package'"
        else
            print_color $YELLOW "  uv add '$package'"
        fi
    done
    print_color $YELLOW "========================================"
    print_color $YELLOW "Use without -d/--dry-run to actually add packages"
    exit 0
fi

# Add packages using uv add
print_color $GREEN "Adding packages using uv add..."
print_color $GREEN "================================"

SUCCESS_COUNT=0
FAILED_COUNT=0

for package in "${packages[@]}"; do
    # Build uv add command
    UV_ADD_CMD="uv add"
    
    if [ "$DEV_DEPS" = true ]; then
        UV_ADD_CMD="$UV_ADD_CMD --dev"
    fi
    
    UV_ADD_CMD="$UV_ADD_CMD '$package'"
    
    print_color $BLUE "Adding: $package"
    
    if [ "$VERBOSE" = true ]; then
        print_color $BLUE "Command: $UV_ADD_CMD"
    fi
    
    # Execute the command
    if eval "$UV_ADD_CMD"; then
        print_color $GREEN "  ✅ Successfully added: $package"
        ((SUCCESS_COUNT++))
    else
        print_color $RED "  ❌ Failed to add: $package"
        failed_packages+=("$package")
        ((FAILED_COUNT++))
    fi
    
    echo # Empty line for readability
done

# Show final results
print_color $GREEN "================================"
print_color $GREEN "📊 Summary:"
print_color $GREEN "  Total packages: $TOTAL_PACKAGES"
print_color $GREEN "  Successfully added: $SUCCESS_COUNT"

if [ $FAILED_COUNT -gt 0 ]; then
    print_color $RED "  Failed to add: $FAILED_COUNT"
    print_color $RED "  Failed packages:"
    for failed_package in "${failed_packages[@]}"; do
        print_color $RED "    - $failed_package"
    done
fi

print_color $BLUE "Installation completed at: $(date)"

# Sync dependencies
if [ $SUCCESS_COUNT -gt 0 ] && [ "$DRY_RUN" = false ]; then
    print_color $BLUE "Syncing dependencies..."
    if uv sync; then
        print_color $GREEN "✅ Dependencies synced successfully!"
    else
        print_color $YELLOW "⚠️  Warning: Failed to sync dependencies. You may need to run 'uv sync' manually."
    fi
fi

# Exit with appropriate code
if [ $FAILED_COUNT -gt 0 ]; then
    exit 1
else
    exit 0
fi