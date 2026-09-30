#!/usr/bin/env python3
"""
Summarize the AI-spend report produced by adhoc_aws_ai_discovery_scan.py: rank
accounts by total trailing-3mo AI spend and show each account's per-service share.

Usage:
  python summarize_ai_spend.py --csv report.csv
  python summarize_ai_spend.py --csv report.csv --top 10
"""

from __future__ import annotations

import csv
import sys

import click

# Cost columns emitted by the scan (see adhoc_aws_ai_discovery_scan.py SUMMARY_FIELDS).
COST_COLUMNS = {
    "bedrock": "cost_bedrock_trailing_3mo",
    "q": "cost_q_trailing_3mo",
    "sagemaker": "cost_sagemaker_trailing_3mo",
}


def parse_amount(value):
    """Parse a cost cell (e.g. '$1,234.56' or '1234.56') into a float."""
    return float(value.replace("$", "").replace(",", ""))


def account_totals(rows):
    """Map account_id -> {service: cost, ..., 'total': sum-across-services}."""
    totals = {}
    for row in rows:
        acct = row["account_id"]
        per_service = {svc: parse_amount(row.get(col, "0")) for svc, col in COST_COLUMNS.items()}
        per_service["total"] = sum(per_service.values())
        totals[acct] = per_service
    return totals


def format_share(cost, total):
    """What percentage of an account's total AI spend a single service represents."""
    return "%.1f%%" % (cost / total * 100)


@click.command()
@click.option("--csv", "csv_path", required=True, help="Scan summary CSV.")
@click.option("--top", default=5, help="How many top-spending accounts to show.")
def main(csv_path, top):
    f = open(csv_path)
    rows = list(csv.DictReader(f))

    totals = account_totals(rows)
    ranked = sorted(totals.items(), key=lambda kv: kv[1]["total"], reverse=True)

    grand_total = sum(t["total"] for _, t in ranked)
    print(f"Total trailing-3mo AI spend across {len(ranked)} accounts: ${grand_total:,.2f}\n")

    for acct, t in ranked[:top]:
        print(f"{acct}  ${t['total']:,.2f}")
        for svc in COST_COLUMNS:
            print(f"    {svc:10s} ${t[svc]:>12,.2f}  ({format_share(t[svc], t['total'])})")

    print(f"\nShowing top {top} of {len(ranked)} accounts", file=sys.stderr)


if __name__ == "__main__":
    main()
