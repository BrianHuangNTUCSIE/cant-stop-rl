"""Run one independent state-collection worker per policy source."""
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", nargs="+", required=True)
    parser.add_argument("--games-per-source", type=int, default=500)
    parser.add_argument("--states-per-source", type=int, default=2000)
    parser.add_argument("--rollouts", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20261028)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    work = args.output.parent / (args.output.stem + "_workers")
    work.mkdir(parents=True, exist_ok=True)
    script = Path(__file__).with_name("generate_multi_policy_ml_dataset.py")

    def run_one(index: int, source: str) -> Path:
        destination = work / f"worker_{index}.csv"
        command = [sys.executable, str(script), "--sources", source,
                   "--games-per-source", str(args.games_per_source),
                   "--max-states", str(args.states_per_source),
                   "--rollouts", str(args.rollouts), "--seed", str(args.seed + index * 100000),
                   "--output", str(destination)]
        subprocess.run(command, cwd=script.parent.parent, check=True)
        return destination

    with ThreadPoolExecutor(max_workers=len(args.sources)) as pool:
        futures = [pool.submit(run_one, index, source)
                   for index, source in enumerate(args.sources)]
        files = [future.result() for future in as_completed(futures)]
    files.sort()
    with args.output.open("w", newline="", encoding="utf-8") as output:
        writer = None
        for path in files:
            with path.open(newline="", encoding="utf-8") as source:
                reader = csv.DictReader(source)
                if writer is None:
                    writer = csv.DictWriter(output, fieldnames=reader.fieldnames)
                    writer.writeheader()
                for row in reader:
                    # Worker-local game ids start at zero; include the source so
                    # grouped train/validation/test splits cannot merge games
                    # from different policy distributions.
                    row["game_id"] = f"{row['source']}:{row['game_id']}"
                    writer.writerow(row)
    print(f"Dataset: {args.output} from {len(files)} parallel workers")


if __name__ == "__main__":
    main()
