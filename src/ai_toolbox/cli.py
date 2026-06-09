"""Minimal CLI for ai_toolbox.

Provides an `add` subcommand for simple demonstration purposes.
"""
import argparse
from .core import add


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="ai-toolbox", description="ai_toolbox CLI")
    sub = parser.add_subparsers(dest="cmd")

    p_add = sub.add_parser("add", help="Add two numbers")
    p_add.add_argument("a", type=float)
    p_add.add_argument("b", type=float)

    args = parser.parse_args(argv)

    if args.cmd == "add":
        result = add(args.a, args.b)
        print(result)
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
