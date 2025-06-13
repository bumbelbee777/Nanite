@echo off
echo Cleaning cache files...

if exist .ruff_cache rmdir /s /q .ruff_cache
if exist .mypy_cache rmdir /s /q .mypy_cache

echo Removing Python cache files...
for /r %%d in (__pycache__) do @if exist "%%d" rmdir /s /q "%%d"
for /r %%f in (*.pyc) do @if exist "%%f" del /f /q "%%f"

echo.
echo Cleanup completed!
