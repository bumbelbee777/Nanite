@echo off
setlocal enabledelayedexpansion

:: Parse command line arguments
set FIX=false
set FIX_UNSAFE=false

:parse_args
if "%~1"=="" goto :end_parse
if "%~1"=="--fix" (
    set FIX=true
    shift
    goto :parse_args
)
if "%~1"=="--fix-unsafe" (
    set FIX=true
    set FIX_UNSAFE=true
    shift
    goto :parse_args
)
shift
goto :parse_args
:end_parse

echo Running linters...

:: Run black
if "%FIX%"=="true" (
    echo Running black with auto-fix...
    black .
) else (
    echo Running black...
    black . --check
)

:: Run isort
if "%FIX%"=="true" (
    echo Running isort with auto-fix...
    isort .
) else (
    echo Running isort...
    isort . --check-only
)

:: Run mypy
echo Running mypy...
mypy .

:: Run ruff
if "%FIX%"=="true" (
    echo Running ruff with auto-fix...
    if "%FIX_UNSAFE%"=="true" (
        ruff check . --fix --unsafe-fixes
    ) else (
        ruff check . --fix
    )
) else (
    echo Running ruff...
    ruff check .
)

echo Linting complete!
