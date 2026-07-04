"""Regenerate the golden snapshot the regression test pins against.

Run this deliberately whenever an *intended* change to the engine shifts the
numbers, then eyeball the git diff of `golden.json` before committing:

    python3 tests/gen_golden.py

It is not run by CI — CI only *checks* against the committed golden.json.
"""
import json
import os

from _snapshot import compute_snapshot

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "golden.json")


def main():
    snap = compute_snapshot()
    with open(OUT, "w") as fh:
        json.dump(snap, fh, indent=2, sort_keys=True)
        fh.write("\n")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
