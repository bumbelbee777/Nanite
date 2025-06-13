@echo off
echo Installing pre-commit hooks...
pre-commit install

echo.
echo Installing development dependencies...
pip install ruff mypy black pre-commit

echo.
echo Setup completed! 