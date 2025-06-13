#!/bin/bash

echo "Cleaning cache files..."

rm -rf .ruff_cache .mypy_cache

echo "Removing Python cache files..."
find . -type d -name "__pycache__" -exec rm -rf {} +
find . -type f -name "*.pyc" -delete

echo
echo "Cleanup completed!"
