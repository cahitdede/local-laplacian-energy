#!/usr/bin/env python3
"""Command line entry point:  python run_all.py [KEY=value ...]

Examples
    python run_all.py
    python run_all.py MAX_ORDER=10
    python run_all.py SECTIONS='["sec4"]' KRR_R_MAX=1000
"""
import ast
import sys

from lle.run import run_all, DEFAULTS

if __name__ == "__main__":
    over = {}
    for arg in sys.argv[1:]:
        if "=" not in arg:
            sys.exit("expected KEY=value, got %r" % arg)
        k, v = arg.split("=", 1)
        if k not in DEFAULTS:
            sys.exit("unknown parameter %s; valid: %s" % (k, ", ".join(sorted(DEFAULTS))))
        try:
            over[k] = ast.literal_eval(v)
        except Exception:
            over[k] = v
    run_all(**over)
