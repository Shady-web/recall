#!/usr/bin/env python3
"""Assert a pytest run actually executed the suite, not just skipped it.

The database-backed tests skip (rather than fail) when no cluster is reachable,
which keeps ``pytest`` usable with no database present — but it means a broken
connection produces a green run that tested almost nothing. ``RECALL_REQUIRE_CRDB=1``
closes that hole at collection time; this script closes it again at the *result*
level, by checking the numbers a human would otherwise have to eyeball.

It exists because this project shipped that exact failure: a 3-second connect
timeout in ``tests/conftest.py`` silently skipped every database-backed test
while CI reported success.

Usage:
    python scripts/check_ci_results.py pytest-report.xml \\
        --min-passed 131 --max-skipped 2
"""

from __future__ import annotations

import argparse
import sys
import xml.etree.ElementTree as ET


def summarize(report_path: str) -> dict[str, int]:
    """Read pytest's JUnit XML into total/failed/errors/skipped/passed counts."""
    root = ET.parse(report_path).getroot()
    # pytest writes <testsuites><testsuite .../></testsuites>; older versions
    # emit a bare <testsuite>. Accept either.
    suite = root if root.tag == "testsuite" else root.find("testsuite")
    if suite is None:
        raise SystemExit(f"{report_path}: no <testsuite> element found")

    total = int(suite.get("tests", 0))
    failures = int(suite.get("failures", 0))
    errors = int(suite.get("errors", 0))
    skipped = int(suite.get("skipped", 0))
    return {
        "total": total,
        "failed": failures,
        "errors": errors,
        "skipped": skipped,
        "passed": total - failures - errors - skipped,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", help="path to pytest's --junitxml output")
    parser.add_argument(
        "--min-passed",
        type=int,
        required=True,
        help="fail if fewer tests passed than this (catches a silently shrunk suite)",
    )
    parser.add_argument(
        "--max-skipped",
        type=int,
        required=True,
        help="fail if more tests skipped than this (catches an unreachable cluster)",
    )
    args = parser.parse_args()

    counts = summarize(args.report)
    print(
        "pytest: {passed} passed, {failed} failed, {errors} errors, "
        "{skipped} skipped ({total} total)".format(**counts)
    )

    problems: list[str] = []
    if counts["passed"] < args.min_passed:
        problems.append(
            f"only {counts['passed']} tests passed, expected at least "
            f"{args.min_passed}. Either the suite shrank, or the database-backed "
            f"tests did not run."
        )
    if counts["skipped"] > args.max_skipped:
        problems.append(
            f"{counts['skipped']} tests skipped, expected at most "
            f"{args.max_skipped}. The usual cause is an unreachable CockroachDB, "
            f"which makes the database-backed tests skip instead of fail."
        )
    if counts["failed"] or counts["errors"]:
        problems.append(
            f"{counts['failed']} failed and {counts['errors']} errored."
        )

    if problems:
        print("\nCI result check FAILED:", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1

    print(
        f"OK: at least {args.min_passed} passed and at most "
        f"{args.max_skipped} skipped."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
