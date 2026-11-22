import argparse
import json
from pathlib import Path
import pandas as pd

def normalize(records):
    mapping = json.loads(Path(__file__).with_name("delegations.json").read_text())
    records = records.loc[records["Year"] >= 1948].copy()
    names = records.NOC.str.encode("ascii", errors="ignore").str.decode("ascii").str.strip()
    records["shortNOC"] = names.map(mapping).fillna("Unknown")
    return records

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", type=Path, nargs="?")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    data = pd.read_csv(args.csv) if args.csv else pd.DataFrame({"Year": [1936, 1948, 2024], "NOC": ["Germany", "France", "Japan"]})
    result = normalize(data)
    if args.csv is None:
        assert result.shortNOC.tolist() == ["FRA", "JPN"]
    if args.output:
        result.to_csv(args.output, index=False)
    print(result.head().to_string(index=False))
