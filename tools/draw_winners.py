#!/usr/bin/env python3
"""
draw_winners.py — pick the prize-draw winners from the Google-Form email export.

In the Form's response Sheet:  File -> Download -> Comma-separated values (.csv)
Then:
    python tools/draw_winners.py giveaway_responses.csv
    python tools/draw_winners.py giveaway_responses.csv --n 3 --seed 20260701

Dedups emails (case-insensitive) and draws N winners. Pass a --seed to make the draw
reproducible/auditable (anyone re-running with the same file + seed gets the same winners).
The email list never touches the study data — this is a standalone, throwaway script.
"""
import csv, sys, argparse, random


def find_email_col(fieldnames):
    for f in fieldnames:
        if "mail" in f.lower():
            return f
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", help="CSV exported from the Google-Form response sheet")
    ap.add_argument("--n", type=int, default=3, help="number of winners (default 3)")
    ap.add_argument("--seed", type=int, default=None,
                    help="random seed — set one for a reproducible, auditable draw")
    ap.add_argument("--col", default=None, help="email column name (auto-detected if omitted)")
    a = ap.parse_args()

    rows = list(csv.DictReader(open(a.csv, encoding="utf-8-sig")))
    if not rows:
        sys.exit("no rows in CSV")
    col = a.col or find_email_col(rows[0].keys())
    if not col:
        sys.exit(f"no email column found. Columns: {list(rows[0].keys())}\nPass --col \"<name>\".")

    emails, seen = [], set()
    for r in rows:
        raw = (r.get(col) or "").strip()
        key = raw.lower()
        if "@" in key and key not in seen:
            seen.add(key); emails.append(raw)

    print(f"{len(rows)} entries -> {len(emails)} unique valid email(s)  (column: '{col}')")
    if len(emails) < a.n:
        print(f"WARNING: only {len(emails)} unique emails (< {a.n}); drawing all of them.")

    rng = random.Random(a.seed)
    winners = rng.sample(emails, min(a.n, len(emails)))
    print(f"\nseed = {a.seed}   (re-run with the same seed to reproduce this draw)")
    print(f"--- {len(winners)} WINNER(S) ---")
    for i, w in enumerate(winners, 1):
        print(f"  {i}. {w}")


if __name__ == "__main__":
    main()
