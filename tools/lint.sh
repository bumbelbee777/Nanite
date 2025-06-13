#!/bin/bash

# Parse command line arguments
FIX=false
FIX_UNSAFE=false

for arg in "$@"; do
    case $arg in
        --fix)
            FIX=true
            shift
            ;;
        --fix-unsafe)
            FIX=true
            FIX_UNSAFE=true
            shift
            ;;
    esac
done

# Run linters
echo "Running linters..."

# Run black
if [ "$FIX" = true ]; then
    echo "Running black with auto-fix..."
    black .
else
    echo "Running black..."
    black . --check
fi

# Run isort
if [ "$FIX" = true ]; then
    echo "Running isort with auto-fix..."
    isort .
else
    echo "Running isort..."
    isort . --check-only
fi

# Run mypy
echo "Running mypy..."
mypy .

# Run ruff
if [ "$FIX" = true ]; then
    echo "Running ruff with auto-fix..."
    if [ "$FIX_UNSAFE" = true ]; then
        ruff check . --fix --unsafe-fixes
    else
        ruff check . --fix
    fi
else
    echo "Running ruff..."
    ruff check .
fi

echo "Linting complete!"
