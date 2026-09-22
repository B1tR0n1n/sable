"""Make `import console...` work from any cwd: the console package lives at
<sable root>/console, and pytest's rootdir is console/ (its pyproject)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]          # the SABLE checkout
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
