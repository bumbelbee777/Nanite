import asyncio

from .cli import PartialSolverCLI


def main():
    """Main entry point."""
    cli = PartialSolverCLI()
    asyncio.run(cli.run())


if __name__ == "__main__":
    main()
