"""Allow `python -m app.cli` from the repository root without copying backend code."""
from pathlib import Path
import os

__path__.append(str(Path(os.path.abspath(__file__)).parents[1] / "backend" / "app"))