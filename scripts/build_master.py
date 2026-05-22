from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from project_utils import MASTER_PATH, save_master_dataframe


def main() -> None:
    master = save_master_dataframe()
    print(f"Saved master dataset to: {MASTER_PATH}")
    print(f"Rows: {len(master)}")
    print(f"States: {master['state'].nunique()}")
    print(f"Date range: {master['date'].min().date()} to {master['date'].max().date()}")


if __name__ == "__main__":
    main()
