#!/bin/bash

echo "Running Ruff linter..."
ruff check .

echo
echo "Running Ruff formatter..."
ruff format .

echo
echo "Running Mypy type checker..."
mypy .

echo
echo "All checks completed!" 