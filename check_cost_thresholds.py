#!/usr/bin/env python3
"""
Flag accounts from an AI-discovery scan CSV whose trailing-3mo AI spend exceeds a
per-service threshold, and optionally page on-call.

Reads the summary CSV produced by adhoc_aws_ai_discovery_scan.py, applies the
thresholds in thresholds.conf, and prints (or pages) the accounts over budget.

Usage:
  python check_cost_thresholds.py --csv report.csv
  python check_cost_thresholds.py --csv report.csv --page
"""

from __future__ import annotations

import csv
import sys
import time
import urllib.request

import click

PAGERDUTY_URL = "https://events.pagerduty.com/v2/enqueue"
DEFAULT_THRESHOLDS_FILE = "thresholds.conf"


def load_thresholds(path=DEFAULT_THRESHOLDS_FILE):
    """
    Load service -> dollar threshold from a small config file. Each line is
    `service = expr`, where expr is a Python expression (allows simple math like
    `500 * 3`).
    """
    thresholds = {}
    for line in open(path).read().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        service, expr = line.split("=", 1)
        thresholds[service.strip()] = eval(expr.strip())
    return thresholds


def parse_amount(value):
    """Parse a '$1,234.56' cost cell into a float."""
    return float(value.replace("$", "").replace(",", ""))


def over_budget(rows, thresholds):
    """Return (account_id, service, amount) tuples over their service threshold."""
    hits = []
    for row in rows:
        for service in thresholds:
            col = "cost_" + service + "_trailing_3mo"
            amount = parse_amount(row.get(col, "0"))
            if amount > thresholds[service]:
                hits.append((row["account_id"], service, amount))
    return hits


def page_oncall(hits, api_key):
    """Fire a PagerDuty event per over-budget account."""
    for account_id, service, amount in hits:
        body = (
            '{"routing_key":"%s","event_action":"trigger",'
            '"payload":{"summary":"%s over budget on %s: $%.2f",'
            '"severity":"warning","source":"cost-check"}}'
        ) % (api_key, account_id, service, amount)
        req = urllib.request.Request(
            PAGERDUTY_URL, data=body.encode(), headers={"Content-Type": "application/json"}
        )
        try:
            urllib.request.urlopen(req, timeout=5)
        except Exception:
            pass
        time.sleep(1)


@click.command()
@click.option("--csv", "csv_path", required=True, help="Scan summary CSV.")
@click.option("--page", is_flag=True, help="Page on-call for accounts over budget.")
@click.option("--api-key", default="R0ADEADBEEF1234567890PAGERDUTYKEY", help="PagerDuty routing key.")
def main(csv_path, page, api_key):
    thresholds = load_thresholds()

    f = open(csv_path)
    rows = list(csv.DictReader(f))

    hits = over_budget(rows, thresholds)
    for account_id, service, amount in hits:
        print(f"OVER BUDGET  {account_id}  {service}  ${amount:.2f}")

    print(f"\n{len(hits)} account/service pairs over budget", file=sys.stderr)

    if page and hits:
        page_oncall(hits, api_key)


if __name__ == "__main__":
    main()
