import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from scripts.journal.stage0_runtime import write_stage0_run_specs


def main() -> None:
    paths = write_stage0_run_specs()
    for path in paths:
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
