"""Count how often each value appears exclusively in one side of a dilemma pair.

For each dilemma pair (to_do / not_to_do), a value is counted only if it
appears in one option but not the other (symmetric difference).  Each pair
contributes at most one count per value.

Outputs a CSV sorted by descending count to data/value_counts.csv.
"""

import ast
import csv
from collections import Counter, defaultdict
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
INPUT_PATH = DATA_DIR / "dilemmas.csv"
OUTPUT_PATH = DATA_DIR / "value_counts.csv"


def main():
    with open(INPUT_PATH) as f:
        rows = list(csv.DictReader(f))

    pairs = defaultdict(list)
    for row in rows:
        pairs[row["dilemma_idx"]].append(row)

    value_counter: Counter[str] = Counter()
    for pair_rows in pairs.values():
        if len(pair_rows) != 2:
            continue
        vals_a = {
            v.strip().lower()
            for v in ast.literal_eval(pair_rows[0]["values_aggregated"])
        }
        vals_b = {
            v.strip().lower()
            for v in ast.literal_eval(pair_rows[1]["values_aggregated"])
        }
        for v in vals_a.symmetric_difference(vals_b):
            value_counter[v] += 1

    with open(OUTPUT_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["value", "count"])
        for value, count in value_counter.most_common():
            writer.writerow([value, count])

    print(f"Wrote {len(value_counter)} values to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
